
import torch
import torch.nn.functional as F



def compute_temporal_loss(raft_model, generated_frames, gt_frames, step=1, scale=1.0):
    """
    generated_frames, gt_frames: [B, M, 3, H, W] 且应已是 [0,1] 归一化
    step : 时间子采样步长（默认隔帧）
    scale: 空间下采样比例（1.0/0.5/0.25 可显著降显存）
    返回：对 generated_frames 可导的标量
    """
    B, M, C, H, W = generated_frames.shape

    # 图内零，避免在无成对帧时断图
    graph_zero = generated_frames.sum() * 0.0
    if M < 2:
        return graph_zero

    # 选取 (t, t+1) 的成对索引（子采样）
    idx = torch.arange(0, M - 1, step, device=generated_frames.device)
    if idx.numel() == 0:
        return graph_zero

    # 取相邻帧并批量化：[(B, |idx|, 3, H, W) -> (B*|idx|, 3, H, W)]
    gen1 = generated_frames[:, idx]          # [B,P,3,H,W]
    gen2 = generated_frames[:, idx + 1]      # [B,P,3,H,W]
    lr1  = gt_frames[:, idx]
    lr2  = gt_frames[:, idx + 1]
    P = gen1.shape[1]

    gen1 = gen1.reshape(B * P, C, H, W)
    gen2 = gen2.reshape(B * P, C, H, W)
    lr1  = lr1.reshape(B * P, C, H, W)
    lr2  = lr2.reshape(B * P, C, H, W)

    # 可选：下采样到更小分辨率喂给 RAFT，显著省显存
    if scale != 1.0:
        new_hw = (max(1, int(H * scale)), max(1, int(W * scale)))
        gen1 = F.interpolate(gen1, size=new_hw, mode="bilinear", align_corners=False)
        gen2 = F.interpolate(gen2, size=new_hw, mode="bilinear", align_corners=False)
        lr1  = F.interpolate(lr1,  size=new_hw, mode="bilinear", align_corners=False)
        lr2  = F.interpolate(lr2,  size=new_hw, mode="bilinear", align_corners=False)



    # gen分支（gen）可导；gt分支（lr）不需要梯度
    flow_gen = raft_model(gen1, gen2)[-1]          # 和外层 autocast 保持一致精度
    with torch.no_grad():
        flow_gt  = raft_model(lr1,  lr2)[-1]

    # 直接在当前尺度上做 MSE
    L = F.mse_loss(flow_gen, flow_gt, reduction="mean")

    # 返回 fp32 标量更稳（不影响显存）
    return L.float()




@torch.no_grad()
def _resize_pair(x, size_hw):
    # 工具：双线性 resize（用于输入给 RAFT 的 LR 帧）
    return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)

def latent_warp_flow_loss(
    raft_model,
    pred_x0_latents_f32: torch.Tensor,  # [B, T, C_lat, h_z, w_z]  (来自 v->x0 的 fp32 latent)
    frames_pixels: torch.Tensor,            # [B, T, 3, H, W]          (LR 或 HR 帧，范围[-1,1] 或 [0,1])
    step: int = 2,                      # 时间子采样：隔帧可显著省算力/显存
    raft_scale: float = 1.0             # 在缩小分辨率上跑 RAFT（如 0.5 或 0.25），会自动做位移缩放修正
) -> torch.Tensor:
    """
    基于 RAFT 的 teacher flow（pixel 域，no_grad）→ warp latent（可导）→ MSE
    返回：标量 loss（对 pred_x0_latents_f32 可导；不穿 VAE）
    """
    device = pred_x0_latents_f32.device
    B, T, C_lat, h_z, w_z = pred_x0_latents_f32.shape
    if T < 2:
        return pred_x0_latents_f32.sum() * 0.0  # graph zero

    # 0) 规范化到 [0,1] 给 RAFT 用
    #    注意：不建图 —— 只当 teacher
    imgs = (frames_pixels.float().clamp(-1, 1) + 1.0) / 2.0  # [B,T,3,H,W]
    _, _, H, W = imgs.shape

    # 1) 预先构建 latent 基础网格（对齐 grid_sample 的 align_corners=True）
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h_z, device=device),
        torch.linspace(-1, 1, w_z, device=device),
        indexing="ij"
    )
    base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)  # [1,h_z,w_z,2]

    loss = 0.0
    pairs = 0

    # 2) 逐对时间步计算（省显存）
    for t in range(0, T - 1, step):
        # 2.1) 取相邻帧；可选：先下采样到更小分辨率跑 RAFT
        with torch.no_grad():
            if raft_scale != 1.0:
                Hs = max(1, int(H * raft_scale))
                Ws = max(1, int(W * raft_scale))
                f1 = _resize_pair(imgs[:, t],   (Hs, Ws))  # [B,3,Hs,Ws]
                f2 = _resize_pair(imgs[:, t+1], (Hs, Ws))
                flow = raft_model(f1, f2)[-1]                    # [B,2,Hs,Ws] 单位=小图像素
                # 直接插值到 latent 尺度
                flow = F.interpolate(flow, size=(h_z, w_z), mode="bilinear", align_corners=False)  # [B,2,h_z,w_z]
                # 将“小图像素位移”换算成“latent 像素位移”
                # 1个小图像素 在 x 方向 ≈ wz/Ws 个 latent 像素；y 方向 ≈ hz/Hs
                flow[:, 0] *= (w_z / float(Ws))
                flow[:, 1] *= (h_z / float(Hs))
            else:
                f1 = imgs[:, t]   # [B,3,H,W]
                f2 = imgs[:, t+1]
                flow = raft_model(f1, f2)[-1]                                  # [B,2,H,W] 单位=原图像素
                flow = F.interpolate(flow, size=(h_z, w_z), mode="bilinear", align_corners=False)
                # 原图像素 → latent 像素：x 乘 wz/W，y 乘 hz/H
                flow[:, 0] *= (w_z / float(W))
                flow[:, 1] *= (h_z / float(H))

        # 2.2) latent 像素位移 → grid_sample 需要的归一化坐标位移（align_corners=True）
        # Δx_norm = 2*dx/(w_z-1), Δy_norm = 2*dy/(h_z-1)
        # w_z/h_z 可能为1，做个保护
        wx = max(w_z - 1, 1)
        hy = max(h_z - 1, 1)
        dx_norm = (2.0 * flow[:, 0]) / float(wx)  # [B,h_z,w_z]
        dy_norm = (2.0 * flow[:, 1]) / float(hy)  # [B,h_z,w_z]
        grid = (base_grid + torch.stack([dx_norm, dy_norm], dim=-1).permute(0, 2, 3, 1)).clamp(-1, 1)  # [B,h_z,w_z,2]

        # 2.3) warp 第 t 帧的 latent（★可导），对齐到 t+1
        z_t   = pred_x0_latents_f32[:, t]     # [B,C_lat,h_z,w_z]
        z_t1  = pred_x0_latents_f32[:, t+1]
        z_t_w = F.grid_sample(
            z_t, grid, mode="bilinear", padding_mode="border", align_corners=True
        )  # [B,C_lat,h_z,w_z]

        loss = loss + F.mse_loss(z_t_w, z_t1, reduction="mean")
        pairs += 1

    if pairs == 0:
        return pred_x0_latents_f32.sum() * 0.0

    return (loss / pairs).float()
