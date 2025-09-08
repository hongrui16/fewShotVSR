import torch
import random

class DummyDataset(torch.utils.data.Dataset):
    """
    Yields tuples (lr_frames, hd_frames, sparse_indices) with shapes:
      lr_frames:  [1, T, 3, H, W]
      hd_frames:  [1, N, 3, H, W]  (N in {0,1,2})
      sparse_indices: [1, N] (long)
    """
    def __init__(self, T=14, H=256, W=256,
                 probs=(0.4, 0.3, 0.3), device="cpu", seed=0):
        self.nums = 2000
        self.T, self.H, self.W = T, H, W
        self.device = device
        self.probs = probs
        
        # 固定 HD 槽位索引 [0, (T -1)//2]（T=14 -> [0,6]；若负则回退 0）, （T=25 -> [0,12]；若负则回退 0）
        mid = max(0, (self.T - 1) // 2 )
        self.sparse_indices = torch.tensor([0, mid], dtype=torch.long)  # [2]
        
        self.height = H
        self.width = W

    def __len__(self):
        return self.nums

    def __iter__(self):
        """
        Yields:
        lr_frames:    [T, C, H, W], 值域 [-1, 1]
        hd_frames:    [2, C, H, W]  固定两个候选槽位（第0帧 & T//2-1）
        mask:         [2]           逐样本启用的槽位（0/1/2）
        """
        device = self.device
        C, T, H, W =3, self.T, self.H, self.W
        assert T >= 2, "T must be >= 2 to use indices [0, T//2 - 1 ]"

        # 概率 40%/30%/30% -> 启用 0/1/2 个 HD 槽位
        probs = torch.tensor([0.4, 0.3, 0.3], device=device)
        cat = torch.distributions.Categorical(probs=probs)

        # 固定的稀疏时间索引：第0帧 & 中间帧
        N = 2


        # LR video in [0, 1]
        lr_frames = torch.rand(T, C, H, W, device=device)

        # 用固定索引从 LR 里抽出候选 HD 槽位
        idx = self.sparse_indices.view(N, 1, 1, 1).expand(N, C, H, W)  # [2,C,H,W]
        hd_frames = lr_frames.gather(dim=0, index=idx)                  # [2,C,H,W]
        # 轻微扰动（可选）
        hd_frames = (hd_frames + 0.01 * torch.randn_like(hd_frames)).clamp(0, 1)

        # 每个样本决定启用 0/1/2 个槽位
        mask = torch.zeros(N, dtype=torch.bool, device=device)
        n_b = int(cat.sample().item())  # 0/1/2
        if n_b > 0:
            mask[:n_b] = True        # 按顺序启用前 n_b 个槽位

        return lr_frames, hd_frames, mask

    def __getitem__(self, idx):
        lr_frames, hd_frames, mask = self.__iter__()
        return lr_frames, hd_frames, mask


if __name__ == "__main__":
    dataset = DummyDataset(device="cpu")
    for i in range(10):
        lr_frames, hd_frames, mask = dataset[i]
        print(f"Sample {i}:")
        print("  lr_frames:", lr_frames.shape, lr_frames.min().item(), lr_frames.max().item())
        print("  hd_frames:", hd_frames.shape, hd_frames.min().item(), hd_frames.max().item())
        print("  mask:", mask, mask.sum().item())