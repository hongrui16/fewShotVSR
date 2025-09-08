import torch
import random

class DummyData:
    """
    Yields tuples (lr_frames, hd_frames, sparse_indices) with shapes:
      lr_frames:  [1, T, 3, H, W]
      hd_frames:  [1, N, 3, H, W]  (N in {0,1,2})
      sparse_indices: [1, N] (long)
    """
    def __init__(self, steps=12, T=14, H=256, W=256,
                 probs=(0.4, 0.3, 0.3), device="cuda", seed=0):
        self.steps = steps
        self.T, self.H, self.W = T, H, W
        self.B = 2
        self.device = device
        self.probs = probs
        random.seed(seed)
        torch.manual_seed(seed)

    def __len__(self):
        return self.steps

    def __iter__(self):
        """
        Yields:
        lr_frames:    [B, T, C, H, W], 值域 [-1, 1]
        hd_frames:    [B, 2, C, H, W]  固定两个候选槽位（第0帧 & 中间帧）
        mask:         [B, 2]           逐样本启用的槽位（0/1/2）
        """
        device = self.device
        B, C, T, H, W = self.B, 3, self.T, self.H, self.W
        assert T >= 2, "T must be >= 2 to use indices [0, T//2]"
        steps = getattr(self, "steps", 1)

        # 概率 40%/30%/30% -> 启用 0/1/2 个 HD 槽位
        probs = torch.tensor([0.4, 0.3, 0.3], device=device)
        cat = torch.distributions.Categorical(probs=probs)

        # 固定的稀疏时间索引：第0帧 & 中间帧
        N = 2
        sparse_indices = torch.tensor([0, T // 2], dtype=torch.long, device=device)  # [2]

        for _ in range(steps):
            # LR video in [-1, 1]
            lr_frames = torch.rand(B, T, C, H, W, device=device) * 2 - 1

            # 用固定索引从 LR 里抽出候选 HD 槽位
            idx = sparse_indices.view(1, N, 1, 1, 1).expand(B, N, C, H, W)  # [B,2,C,H,W]
            hd_frames = lr_frames.gather(dim=1, index=idx)                  # [B,2,C,H,W]
            # 轻微扰动（可选）
            hd_frames = (hd_frames + 0.01 * torch.randn_like(hd_frames)).clamp(-1, 1)

            # 每个样本决定启用 0/1/2 个槽位
            mask = torch.zeros(B, N, dtype=torch.bool, device=device)
            for b in range(B):
                n_b = int(cat.sample().item())  # 0/1/2
                if n_b > 0:
                    mask[b, :n_b] = True        # 按顺序启用前 n_b 个槽位

            yield lr_frames, hd_frames, mask
