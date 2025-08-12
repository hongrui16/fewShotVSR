import torch
import random

class DummyFewShotDataset:
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
        self.device = device
        self.probs = probs
        random.seed(seed)
        torch.manual_seed(seed)

    def __len__(self):
        return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            # sample N in {0,1,2} with probs
            N = random.choices([0, 1, 2], weights=self.probs, k=1)[0]

            # B=1 for simplicity
            B, C, T, H, W = 1, 3, self.T, self.H, self.W

            # LR video in [-1, 1]
            lr_frames = torch.rand(B, T, C, H, W, device=self.device) * 2 - 1

            if N == 0:
                # empty HD and indices
                hd_frames = torch.empty(B, 0, C, H, W, device=self.device)
                sparse_indices = torch.empty(B, 0, dtype=torch.long, device=self.device)
            else:
                # choose distinct indices
                idx_list = sorted(random.sample(range(T), N))
                sparse_indices = torch.tensor(idx_list, dtype=torch.long, device=self.device).unsqueeze(0)  # [1, N]

                # 方式一：逐帧拼接
                hd_list = []
                for t_idx in idx_list:
                    hd_list.append(lr_frames[:, t_idx:t_idx+1, :, :, :])  # [1,1,3,H,W]
                hd_frames = torch.cat(hd_list, dim=1)  # [1,N,3,H,W]

                # 可选：给 HD 加点微小扰动，避免完全相同
                hd_frames = (hd_frames + 0.01 * torch.randn_like(hd_frames)).clamp(-1, 1)

            yield lr_frames, hd_frames, sparse_indices
