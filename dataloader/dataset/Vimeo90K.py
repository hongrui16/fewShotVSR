import os, glob, random
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np


# ===================== 工具函数 =====================

def build_vimeo90k_clips(sequences_root: str, split_txt: Optional[str] = None) -> List[str]:
    """
    构建 Vimeo-90K(septuplet) 的 clip 列表。
    - 官方目录: {sequences_root}/{vid_id}/{clip_id}/im1.png..im7.png
    - split_txt: 每行一个 "00001/0001" 这样的相对路径
    返回: 每个元素是完整的 clip 目录路径（包含 im1..im7）
    """
    clips = []
    if split_txt is not None and os.path.isfile(split_txt):
        with open(split_txt, 'r') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        for rel in lines:
            clip_dir = os.path.join(sequences_root, rel)
            ims = [os.path.join(clip_dir, f"im{i}.png") for i in range(1, 8)]
            if all(os.path.isfile(p) for p in ims):
                clips.append(clip_dir)
    else:
        lvl1 = sorted([d for d in glob.glob(os.path.join(sequences_root, '*')) if os.path.isdir(d)])
        for d1 in lvl1:
            lvl2 = sorted([d for d in glob.glob(os.path.join(d1, '*')) if os.path.isdir(d)])
            for d2 in lvl2:
                ims = [os.path.join(d2, f"im{i}.png") for i in range(1, 8)]
                if all(os.path.isfile(p) for p in ims):
                    clips.append(d2)
    if len(clips) == 0:
        raise FileNotFoundError("No valid Vimeo-90K clips found. Check sequences_root or split_txt.")
    return clips


def _to_tensor_and_norm(img_np: np.ndarray, to_neg1_pos1: bool) -> torch.Tensor:
    """
    img_np: HxWx3, uint8
    return: [3,H,W], float32 in [-1,1] or [0,1]
    """
    t = torch.from_numpy(img_np).float() / 255.0              # [H,W,3] in [0,1]
    t = t.permute(2, 0, 1).contiguous()                       # [3,H,W]
    return t * 2 - 1 if to_neg1_pos1 else t


def _pil_bicubic_resize(img_np: np.ndarray, out_wh: Tuple[int, int]) -> np.ndarray:
    """np.uint8 -> bicubic -> np.uint8"""
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


def mirror_pad_7_to_14(frames_7: List[np.ndarray]) -> List[np.ndarray]:
    """[1..7, 7..1] 的镜像扩展"""
    return frames_7 + frames_7[::-1]


# ===================== Dataset =====================

class Vimeo7to14Dataset(Dataset):
    """
    读取 Vimeo-90K(septuplet) 的 7 帧 clip，并在 __getitem__:
      - (train) 可选随机裁剪/水平翻转；(val/test) 不增广（可选 center crop 或不裁剪）
      - 7 -> 14 镜像扩展
      - 由 HR 生成 LR (×4 bicubic)
      - 归一化到 [-1,1]（可改为 [0,1]）
      - 生成 HD 槽位: 固定索引 [0, T//2 - 1]  (T=14 -> [0, 6])
      - 槽位掩码: 40%/30%/30% 启用 0/1/2 个槽位（从前往后启用）
    返回:
      LR:            [T, 3, h, w]   in [-1,1]
      HR:            [T, 3, H, W]   in [-1,1]
      hd_frames:     [2, 3, h, w]   从 LR gather
      hd_mask:       [2] bool
      sparse_indices:[2] long
    """
    def __init__(
        self,
        sequences_root: str,            # .../vimeo_septuplet/sequences
        split: str = "train",           # "train" | "val" | "test"
        split_txt: Optional[str] = None,# e.g. .../sep_trainlist.txt
        scale: int = 4,
        crop_size_hr: Optional[int] = 256,   # HR 裁剪尺寸；val/test 若不想裁剪设为 None
        to_neg1_pos1: bool = True,
        # HD 槽位
        use_hd_noise: bool = True,
        hd_noise_std: float = 0.01,
        hd_probs: Tuple[float, float, float] = (0.4, 0.3, 0.3),  # 0/1/2 槽位启用概率
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

        self.T = 14
        self.C = 3

        # 固定 HD 槽位索引 [0, (T -1)//2]（T=14 -> [0,6]；若负则回退 0）
        mid = max(0, (self.T - 1) // 2 )
        self.sparse_indices = torch.tensor([0, mid], dtype=torch.long)  # [2]
        
        
        # 索引 Vimeo-90K clips
        self.clips = build_vimeo90k_clips(sequences_root, split_txt)

    def __len__(self):
        return len(self.clips)

    def _sample_hd_mask(self) -> torch.Tensor:
        n_use = int(self.cat.sample().item())  # 0 / 1 / 2
        m = torch.zeros(2, dtype=torch.bool)
        if n_use > 0:
            m[:n_use] = True
        return m

    def _load_hr7(self, clip_dir: str) -> List[np.ndarray]:
        # 固定文件名顺序，避免排序歧义
        paths = [os.path.join(clip_dir, f"im{i}.png") for i in range(1, 8)]
        imgs = [np.array(Image.open(p).convert("RGB")) for p in paths]  # list of HxWx3, uint8
        return imgs

    def _apply_transforms_hr(self, imgs: List[np.ndarray]) -> List[np.ndarray]:
        """
        - train: 随机裁剪 + 随机水平翻转
        - val/test: 若 crop_size_hr 非 None -> 居中裁剪；否则不裁剪
        """
        if self.crop_size_hr is None:
            # 不裁剪
            if self.split == "train":
                # 训练时若不裁剪，也可仅做随机水平翻转
                if random.random() < 0.5:
                    imgs = [img[:, ::-1, :].copy() for img in imgs]
            return imgs

        # pad 到至少 crop_size（避免短边不足）
        imgs = [_pad_to_min_size_edge(img, self.crop_size_hr, self.crop_size_hr) for img in imgs]
        H, W = imgs[0].shape[:2]

        if self.split == "train":
            # 随机裁剪 + 随机水平翻转
            top = random.randint(0, H - self.crop_size_hr)
            left = random.randint(0, W - self.crop_size_hr)
            imgs = [img[top:top+self.crop_size_hr, left:left+self.crop_size_hr, :].copy() for img in imgs]
            if random.random() < 0.5:
                imgs = [img[:, ::-1, :].copy() for img in imgs]
        else:
            # val/test: 中心裁剪（稳定、无随机性）
            imgs = [_center_crop_np(img, self.crop_size_hr).copy() for img in imgs]

        return imgs

    def __getitem__(self, idx):
        clip_dir = self.clips[idx]

        # 1) 读 HR(7) -> 变换（受 split 控制）
        HR7 = self._load_hr7(clip_dir)                    # list len=7
        HR7 = self._apply_transforms_hr(HR7)

        # 2) 7 -> 14 镜像扩展
        HR14 = mirror_pad_7_to_14(HR7)                    # len=14

        # 3) 由 HR 生成 LR（×scale bicubic）
        h_hr, w_hr = HR14[0].shape[:2]
        h_lr, w_lr = h_hr // self.scale, w_hr // self.scale
        LR14 = [_pil_bicubic_resize(hr, (w_lr, h_lr)) for hr in HR14]

        # 4) 归一化并堆叠
        HR = torch.stack([_to_tensor_and_norm(hr, self.to_neg1_pos1) for hr in HR14], dim=0)  # [14,3,H,W]
        LR = torch.stack([_to_tensor_and_norm(lr, self.to_neg1_pos1) for lr in LR14], dim=0)  # [14,3,h,w]

        

        # 5) 从 LR gather 出 hd_frames: [2,3,h,w]
        #    注意 gather 维度: 时间维在 dim=0，因此 index 需要 [2,3,h,w]
        idx_exp = self.sparse_indices.view(2, 1, 1, 1).expand(2, self.C, LR.shape[-2], LR.shape[-1])
        hd_frames = LR.gather(dim=0, index=idx_exp)  # [2,3,h,w]
        if self.use_hd_noise:
            hd_frames = (hd_frames + self.hd_noise_std * torch.randn_like(hd_frames)).clamp(-1, 1)

        # 7) 槽位掩码（40/30/30 -> 0/1/2 个启用；从前往后）
        hd_mask = self._sample_hd_mask()  # [2] bool

        return LR, HR, hd_frames, hd_mask



if __name__ == "__main__":
    # 简单测试
    ds = Vimeo7to14Dataset(
        sequences_root="/path/to/vimeo_septuplet/sequences",
        split="train",
        split_txt=None,
        crop_size_hr=256,
        scale=4,
        to_neg1_pos1=True,
        use_hd_noise=True,
        hd_noise_std=0.01,
    )
    print(f"Dataset size: {len(ds)}")
    for i in range(3):
        LR, HR, hd_frames, hd_mask, sparse_indices = ds[i]
        print(f"Sample {i}:")
        print(f"  LR: {LR.shape}, HR: {HR.shape}")
        print(f"  hd_frames: {hd_frames.shape}, hd_mask: {hd_mask}, sparse_indices: {sparse_indices}")
        print(f"  LR min/max: {LR.min().item():.3f}/{LR.max().item():.3f}")
        print(f"  HR min/max: {HR.min().item():.3f}/{HR.max().item():.3f}")
        print(f"  hd_frames min/max: {hd_frames.min().item():.3f}/{hd_frames.max().item():.3f}")
        print(f"  hd_mask: {hd_mask}, sparse_indices: {sparse_indices}")
        print()
# ===================== DataLoader 便捷函数 =====================

