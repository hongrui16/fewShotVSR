import os, glob, random
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import torch.nn.functional as F


def build_vimeo90k_clips(sequences_root: str, split_txt: str) -> List[str]:
    """
    split_txt lines look like: 00001/0389
    We map to: <sequences_root>/00001/0389
    """
    clips = []
    with open(split_txt, "r") as f:
        for line in f:
            key = line.strip()
            if not key:
                continue
            clip_dir = os.path.join(sequences_root, key)
            # minimal existence check (im1.png)
            if os.path.exists(os.path.join(clip_dir, "im1.png")):
                clips.append(clip_dir)
    if not clips:
        raise RuntimeError(f"No clips found. Check paths. sequences_root={sequences_root}, split_txt={split_txt}")
    return clips

def _to_tensor_and_norm(img_np: np.ndarray, to_neg1_pos1: bool=False) -> torch.Tensor:
    # img_np: HxWx3 uint8
    t = torch.from_numpy(img_np).permute(2,0,1).float() / 255.0
    return (t * 2 - 1) if to_neg1_pos1 else t

def _pil_bicubic_resize(img_np: np.ndarray, size_wh: Tuple[int,int]) -> np.ndarray:
    return np.array(Image.fromarray(img_np).resize(size_wh, Image.BICUBIC))

def _center_crop_np(img: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    H, W = img.shape[:2]
    top = max(0, (H - crop_h)//2)
    left = max(0, (W - crop_w)//2)
    return img[top:top+crop_h, left:left+crop_w, :]


def _pad_to_min_size_edge(img: np.ndarray, min_h: int, min_w: int) -> np.ndarray:
    H, W = img.shape[:2]
    pad_h = max(0, min_h - H)
    pad_w = max(0, min_w - W)
    if pad_h == 0 and pad_w == 0:
        return img
    out = np.zeros((H+pad_h, W+pad_w, 3), dtype=img.dtype)
    out[:H, :W] = img
    return out


def mirror_pad_7_to_14(frames_7: List[np.ndarray]) -> List[np.ndarray]:
    """[1..7, 7..1] 的镜像扩展"""
    return frames_7 + frames_7[::-1]

IDX_7TO14 = torch.tensor([0,1,2,3,4,5,6,5,4,3,2,1,0,1], dtype=torch.long)


# ===================== Dataset =====================

class Vimeo7to14Dataset(Dataset):
    """
    Vimeo-90K (septuplet):
      - reads 7-frame clip (im1.png..im7.png)
      - optional crop/flip for train; val/test: center crop or no crop
      - mirror-expand 7 -> 14
      - create LR by downscaling HR (×scale, bicubic), THEN upsample LR back to HR size for model input
      - return LR_up and HR both at the SAME resolution (e.g., 256x448)
      - gather two HD slots (time indices [0, 6] for T=14) from LR_up
      - random HD-slot mask with probs (0/1/2 slots): 0.4 / 0.3 / 0.3
    Returns:
      LR_up:         [T, 3, H, W] in [-1,1]  (what SVD sees as "LR")
      HR:            [T, 3, H, W] in [-1,1]  (GT)
      hd_frames:     [2, 3, H, W] (from LR_up for conditioning)
      hd_mask:       [2] bool
      sparse_indices:[2] long  (always [0, 6] when T=14)

    """
    def __init__(
        self,
        sequences_root: str,            # .../vimeo_septuplet/sequences
        split: str = "train",           # "train" | "val" | "test"
        split_txt: Optional[str] = None,# e.g. .../sep_trainlist.txt
        scale: int = 4,
        crop_size_hr: Optional[Tuple[int,int]] = None,  # (H,W) or None. Keep None to preserve 256x448.
        to_neg1_pos1: bool = False,
        use_hd_noise: bool = False,
        hd_noise_std: float = 0.01,
        hd_probs: Tuple[float, float, float] = (0.4, 0.3, 0.3),  # 0/1/2 slots enabled
        **kwargs,
    ):
        self.split = split
        self.scale = int(scale)
        self.to_neg1_pos1 = to_neg1_pos1
        self.crop_size_hr = crop_size_hr  # None -> keep native size (e.g., 256x448)

        self.use_hd_noise = use_hd_noise
        self.hd_noise_std = float(hd_noise_std)
        self.hd_probs = torch.tensor(hd_probs, dtype=torch.float32)
        self.cat = torch.distributions.Categorical(self.hd_probs)

        self.T = 14
        self.C = 3

        mid = max(0, (self.T - 1) // 2)   # -> 6
        self.sparse_indices = torch.tensor([0, mid], dtype=torch.long)  # [2]

        if split_txt is None:
            raise ValueError("Please provide split_txt (e.g., trainlist.txt / testlist.txt).")
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
        # train: random crop + hflip; val/test: center crop if crop_size_hr else keep
        if self.crop_size_hr is None:
            # no crop; optional flip on train
            if self.split == "train" and random.random() < 0.5:
                imgs = [img[:, ::-1, :].copy() for img in imgs]
            return imgs

        ch, cw = self.crop_size_hr
        imgs = [_pad_to_min_size_edge(img, ch, cw) for img in imgs]
        H, W = imgs[0].shape[:2]

        if self.split == "train":
            top = random.randint(0, H - ch)
            left = random.randint(0, W - cw)
            imgs = [img[top:top+ch, left:left+cw, :].copy() for img in imgs]
            if random.random() < 0.5:
                imgs = [img[:, ::-1, :].copy() for img in imgs]
        else:
            imgs = [_center_crop_np(img, ch, cw).copy() for img in imgs]
        return imgs


    def __getitem__(self, idx):
        clip_dir = self.clips[idx]

        # 1) 读 7 帧 + 变换（numpy）
        HR7_np = self._load_hr7(clip_dir)
        HR7_np = self._apply_transforms_hr(HR7_np)

        # 2) stack & normalize -> [7,3,H,W] (torch)
        HR7 = torch.stack([_to_tensor_and_norm(x, self.to_neg1_pos1) for x in HR7_np], dim=0)

        # 3) 7 -> 14（tensor 索引，最快）
        idx_map = IDX_7TO14.to(HR7.device)
        HR = HR7.index_select(dim=0, index=idx_map)        # [14,3,H,W]

        # 4) 由 HR 生成 LR：先下采样到 H//scale, W//scale，再上采样回 H,W
        H, W = HR.shape[-2:]
        h_lr, w_lr = H // self.scale, W // self.scale
        LR_down = F.interpolate(HR, size=(h_lr, w_lr), mode="bicubic", align_corners=False)
        LR      = F.interpolate(LR_down, size=(H, W),    mode="bicubic", align_corners=False)
        # 现在 LR 与 HR 同分辨率（如 256x448），满足 SVD/UNet/VAE 的输入需求

        # 5) 从 LR 里取 HD 槽位帧（时间维 dim=0）
        idx2 = self.sparse_indices.to(LR.device)           # [0,6]
        hd_frames = LR.index_select(dim=0, index=idx2)     # [2,3,H,W]
        if self.use_hd_noise:
            hd_frames = (hd_frames + self.hd_noise_std * torch.randn_like(hd_frames)).clamp(-1, 1)

        # 6) 槽位掩码
        hd_mask = self._sample_hd_mask()                   # [2] bool

        return LR, HR, hd_frames, hd_mask




if __name__ == "__main__":
    sequences_root = "/data/vimeo_septuplet/sequences"
    train_txt = "/data/vimeo_septuplet/sep_trainlist.txt"   # 或你的 trainlist.txt
    test_txt  = "/data/vimeo_septuplet/sep_testlist.txt"    # 或你的 testlist.txt

    train_set = Vimeo7to14Dataset(
        sequences_root, split="train", split_txt=train_txt,
        scale=4, crop_size_hr=None,  # 保持 256x448，不裁剪
    )

    test_set = Vimeo7to14Dataset(
        sequences_root, split="test", split_txt=test_txt,
        scale=4, crop_size_hr=None,
    )


    data = next(iter(DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)))
    print([t.shape for t in data])