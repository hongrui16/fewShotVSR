import os, glob, random
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# ---------- 通用工具 ----------
def _to_tensor_and_norm(img_np: np.ndarray, to_neg1_pos1: bool) -> torch.Tensor:
    t = torch.from_numpy(img_np).float() / 255.0   # [H,W,3] in [0,1]
    t = t.permute(2, 0, 1).contiguous()            # [3,H,W]
    return t * 2 - 1 if to_neg1_pos1 else t

def _pil_bicubic_resize(img_np: np.ndarray, out_wh: Tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(img_np)
    pil = pil.resize(out_wh, resample=Image.BICUBIC)
    return np.array(pil)

def _center_crop_np(img_np: np.ndarray, size: int) -> np.ndarray:
    H, W = img_np.shape[:2]
    top = max(0, (H - size) // 2)
    left = max(0, (W - size) // 2)
    return img_np[top:top+size, left:left+size, :]

def _pad_to_min_size_edge(img_np: np.ndarray, min_h: int, min_w: int) -> np.ndarray:
    H, W = img_np.shape[:2]
    pad_h = max(0, min_h - H)
    pad_w = max(0, min_w - W)
    if pad_h == 0 and pad_w == 0:
        return img_np
    return np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')

def _read_png(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))

# ---------- REDS 索引 ----------
def build_reds_sequences(sequences_root: str, seq_list_txt: Optional[str] = None) -> List[str]:
    """
    sequences_root 指向 HR 目录（如 .../REDS/train_sharp 或 valid_sharp/test_sharp）
    每个子目录一个序列（000, 001, ...），内部 00000000.png ... 00000099.png。
    seq_list_txt 可选：若提供，就只取其中列出的序列名（如 REDS4: 000,011,015,020）。
    """
    if seq_list_txt and os.path.isfile(seq_list_txt):
        with open(seq_list_txt, 'r') as f:
            seq_names = [ln.strip() for ln in f.readlines() if ln.strip()]
        seq_dirs = [os.path.join(sequences_root, s) for s in seq_names]
    else:
        seq_dirs = sorted([d for d in glob.glob(os.path.join(sequences_root, '*')) if os.path.isdir(d)])

    valid_dirs = []
    for sd in seq_dirs:
        frames = sorted(glob.glob(os.path.join(sd, '*.png')))
        if len(frames) >= 100:
            valid_dirs.append(sd)
    if len(valid_dirs) == 0:
        raise FileNotFoundError("No valid REDS sequences found.")
    return valid_dirs

def build_sliding_windows(seq_dir: str, window_T: int = 25, stride: int = 25) -> List[List[str]]:
    """从一个序列目录构建滑窗（默认 25 帧窗；训练可设 stride < 25 做重叠）"""
    frame_paths = sorted(glob.glob(os.path.join(seq_dir, '*.png')))
    N = len(frame_paths)
    windows = []
    if N < window_T:
        return windows
    for s in range(0, N - window_T + 1, stride):
        win = frame_paths[s:s + window_T]
        if len(win) == window_T:
            windows.append(win)
    return windows

# ---------- Dataset: REDS -> 25 帧滑窗 ----------
class REDS25WindowDataset(Dataset):
    """
    返回：
      LR:            [T=25, 3, h, w]  in [-1,1]
      HR:            [T=25, 3, H, W]  in [-1,1]
      hd_frames:     [2,   3, h, w]   从 LR gather（若想从 HR 抽，下面一行替换源张量即可）
      hd_mask:       [2]   bool       40%/30%/30% -> 启用 0/1/2 槽位（按顺序启用）
      sparse_indices:[2]   long       固定两个槽位：[0, T//2-1]，25 -> [0, 11]
    """
    def __init__(
        self,
        sequences_root: str,
        split: str = "train",                # "train" | "val" | "test"
        seq_list_txt: Optional[str] = None,  # 可传入 REDS4 列表
        window_T: int = 25,        
        scale: int = 4,
        crop_size_hr: Optional[int] = 256,   # HR 裁剪；val/test 设 None 可用全图
        to_neg1_pos1: bool = True,
        # HD 槽位
        use_hd_noise: bool = True,
        hd_noise_std: float = 0.01,
        hd_probs: Tuple[float, float, float] = (0.4, 0.3, 0.3),
    ):
        assert split in ("train", "val", "test")
        self.split = split
        self.scale = int(scale)
        self.to_neg1_pos1 = to_neg1_pos1
        self.crop_size_hr = crop_size_hr

        self.use_hd_noise = use_hd_noise
        self.hd_noise_std = float(hd_noise_std)
        self.hd_probs = torch.tensor(hd_probs, dtype=torch.float32)
        self.cat = torch.distributions.Categorical(self.hd_probs)

        # stride = 25                       # 训练常用 12 或 8 做重叠；验证/测试用 25
        if split == "train":
            stride = max(8, window_T // 2)
        else:
            stride = window_T

        self.T = int(window_T)
        assert self.T >= 2
        self.stride = int(stride)
        self.C = 3
        
        # 固定 HD 槽位索引 [0, (T -1)//2]（T=14 -> [0,6]；若负则回退 0）, （T=25 -> [0,12]；若负则回退 0）
        mid = max(0, (self.T - 1) // 2 )
        self.sparse_indices = torch.tensor([0, mid], dtype=torch.long)  # [2]

        # 序列与滑窗
        self.seq_dirs = build_reds_sequences(sequences_root, seq_list_txt)
        self.windows: List[List[str]] = []
        for sd in self.seq_dirs:
            self.windows += build_sliding_windows(sd, self.T, self.stride)
        if len(self.windows) == 0:
            raise RuntimeError("No sliding windows were built. Check T/stride or frames layout.")

    def __len__(self):
        return len(self.windows)

    def _sample_hd_mask(self) -> torch.Tensor:
        n_use = int(self.cat.sample().item())  # 0 / 1 / 2
        m = torch.zeros(2, dtype=torch.bool)
        if n_use > 0:
            m[:n_use] = True
        return m

    def _apply_transforms_hr(self, imgs: List[np.ndarray]) -> List[np.ndarray]:
        """
        - train: 随机裁剪 + 随机水平翻转
        - val/test: 若 crop_size_hr 非 None -> 居中裁剪；否则不裁剪
        """
        if self.crop_size_hr is None:
            if self.split == "train" and random.random() < 0.5:
                imgs = [img[:, ::-1, :].copy() for img in imgs]
            return imgs

        imgs = [_pad_to_min_size_edge(img, self.crop_size_hr, self.crop_size_hr) for img in imgs]
        H, W = imgs[0].shape[:2]

        if self.split == "train":
            top = random.randint(0, H - self.crop_size_hr)
            left = random.randint(0, W - self.crop_size_hr)
            imgs = [img[top:top+self.crop_size_hr, left:left+self.crop_size_hr, :].copy() for img in imgs]
            if random.random() < 0.5:
                imgs = [img[:, ::-1, :].copy() for img in imgs]
        else:
            imgs = [_center_crop_np(img, self.crop_size_hr).copy() for img in imgs]

        return imgs

    def __getitem__(self, idx):
        frame_paths = self.windows[idx]                 # len = T
        HR_list = [_read_png(p) for p in frame_paths]   # 读取 HR 全帧

        HR_list = self._apply_transforms_hr(HR_list)    # 变换（按 split 控制）

        # HR -> LR（×scale）
        h_hr, w_hr = HR_list[0].shape[:2]
        h_lr, w_lr = h_hr // self.scale, w_hr // self.scale
        LR_list = [_pil_bicubic_resize(hr, (w_lr, h_lr)) for hr in HR_list]

        # 归一化堆叠
        HR = torch.stack([_to_tensor_and_norm(hr, self.to_neg1_pos1) for hr in HR_list], dim=0)  # [T,3,H,W]
        LR = torch.stack([_to_tensor_and_norm(lr, self.to_neg1_pos1) for lr in LR_list], dim=0)  # [T,3,h,w]

        
        # 从 LR gather 出 hd_frames: [2,3,h,w]（若要从 HR 抽，换成 HR 即可）
        idx_exp = self.sparse_indices.view(2, 1, 1, 1).expand(2, self.C, LR.shape[-2], LR.shape[-1])
        hd_frames = LR.gather(dim=0, index=idx_exp)
        if self.use_hd_noise:
            hd_frames = (hd_frames + self.hd_noise_std * torch.randn_like(hd_frames)).clamp(-1, 1)

        hd_mask = self._sample_hd_mask()  # [2] bool

        return LR, HR, hd_frames, hd_mask



if __name__ == "__main__":
    
    pass