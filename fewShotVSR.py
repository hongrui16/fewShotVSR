import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import StableVideoDiffusionPipeline, DPMSolverMultistepScheduler
from peft import LoraConfig, get_peft_model
from transformers import CLIPVisionModel, CLIPProcessor
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from torchvision.models.optical_flow import raft_small
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# import sys, os
# sys.path.append(os.path.dirname(__file__))  # 把当前目录加入路径

from torch.amp import GradScaler, autocast

from dummy_dataset import DummyFewShotDataset
from loss_func import compute_temporal_loss, latent_warp_flow_loss

class FewShotVideoSRTrainer:
    def __init__(self, device="cpu"):
        self.device = device
        self.motion_bucket_id = 127  # As in SVD, motion_bucket_id is 127
        self.noise_aug_strength = 0.02
        self.num_frames = 14  # For SVD, max frames is 14
        self.embed_dim = 768  # CLIP embed dimension

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.use_latent_warp = True
        print("Using latent warp:", self.use_latent_warp)

        # Load SVD pipeline
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            torch_dtype=torch.float16
        ).to(self.device)


        print(self.pipe.scheduler.config.prediction_type)  # "v_prediction"

        if not hasattr(self.pipe, "do_classifier_free_guidance"):
            self.do_classifier_free_guidance = False
        else:
            self.do_classifier_free_guidance = self.pipe.do_classifier_free_guidance
        print('self.do_classifier_free_guidance', self.do_classifier_free_guidance)

        self.use_adaptive_gate = True

        self.unet_cross_attention_dim = self.pipe.unet.config.cross_attention_dim
        print('self.unet_cross_attention_dim', self.unet_cross_attention_dim)
        

        if self.use_adaptive_gate:
            self.g_mlp = nn.Sequential(
                nn.Linear(3 * self.unet_cross_attention_dim, 128),  # For adaptive gate: input [LR, HD, |diff|] = 3D
                nn.ReLU(),
                nn.Linear(128, 1)  # Output scalar per-frame g
            ).to(self.device)
        else:
            self.hd_gate = nn.Parameter(torch.tensor(-1.0, device= self.device))  # Init low (~0.27) for early stability
        
        self.pos_abs = nn.Embedding(self.num_frames, self.unet_cross_attention_dim).to(self.device)  # Learnable positional encoding
        self.src_type = nn.Embedding(2, self.unet_cross_attention_dim).to(self.device)  # 0: LR, 1: HD
        # self.cross_proj = nn.Linear(self.embed_dim, self.unet_cross_attention_dim).to(self.device)  # Projection if needed
        # self.cond_ln = nn.LayerNorm(self.unet_cross_attention_dim).to(self.device)
        self.T_max = self.num_frames  # 14

        mid = (self.T_max - 1) // 2
        self.fixed_indices = torch.tensor([0, mid], dtype=torch.long, device=self.device)

        self.N = 2  # 槽位数（few-shot 数）


        # Apply LoRA to U-Net
        lora_config = LoraConfig(
            r=8,
            lora_alpha=4,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.1
        )
        self.pipe.unet = get_peft_model(self.pipe.unet, lora_config)
        print("LoRA applied.")


        self.param_type = self.pipe.unet.dtype
        print(f'param_type: {self.param_type}')

        # self.data_type = torch.float32 
        self.data_type = torch.float32

        # self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.amp_dtype = torch.float16

        self.chunk_size = 8
       
        # LPIPS
        # self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(self.device)

        # RAFT for temporal loss
        self.raft = raft_small(pretrained=True).eval().to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.pipe.unet.parameters()), lr=1e-5)

        # Scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler.alphas_cumprod = self.pipe.scheduler.alphas_cumprod.to(self.device)

        self.pipe.unet.enable_gradient_checkpointing()
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("xFormers enabled.")
        except Exception:
            print("xFormers not available. Proceeding without it.")
            pass



        self.pipe.vae.eval().requires_grad_(False)      # 不更新 VAE 权重，但允许梯度对输入生效
        # self.lpips.eval().requires_grad_(False)         # 不更新 LPIPS 权重，但允许对输入回传
        for p in self.raft.parameters():                # RAFT 同理：参数不更新，但允许对输入回传
            p.requires_grad = False
        self.raft.eval()

    

    def get_added_time_ids(self, batch_size=1):
        fps = 7 - 1  # As in SVD, fps-1
        dtype = torch.float16
        num_videos_per_prompt = 1
        do_classifier_free_guidance = False
        added_time_ids = self.pipe._get_add_time_ids(
            fps,
            self.motion_bucket_id,
            self.noise_aug_strength,
            dtype,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance,
        ).to(self.device)
        return added_time_ids
    

    def positional_encoding(self, pos_ids):
        """
        pos_ids: tensor of shape (B, N)

        """
        B, T = pos_ids.shape
        pe = torch.zeros(B, T, self.embed_dim, device=self.device)
        position = pos_ids.float().unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2, device=self.device).float() * 
                            -(torch.log(torch.tensor(10000.0)) / self.embed_dim))
        pe[..., 0::2] = torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(position * div_term)
        return pe


    @torch.no_grad()
    def _encode_frames_clip(self, frames):
        """
        frames: shape [B, T, C, H, W]
        Helper to encode frames with CLIP: [B, T, C, H, W] -> [B, T, D]
        Handles chunking if needed.
        """
        B, T, C, H, W = frames.shape
        flat = frames.reshape(B*T, C, H, W)

        # CLIP 的输入分辨率，224×224
        clip_res = self.pipe.feature_extractor.size if hasattr(self.pipe, "feature_extractor") else 224
        if isinstance(clip_res, dict):  # 兼容 {"shortest_edge": 224} 
            clip_res = clip_res.get("shortest_edge", 224)

        flat = F.interpolate(flat, size=(clip_res, clip_res), mode="bilinear", align_corners=False)



        embeds = self.pipe._encode_image(flat, self.device, 1, False).reshape(B, T, -1)
        return embeds

    def build_image_embeddings(self, lr_frames, hd_frames=None, indices_mask=None,
                               spread_radius=0):
        """
        inputs:
            lr_frames: [B, T, 3, H, W]
            hd_frames: [B, N, 3, H, W]
            indices_mask: [B, N]
        Returns:
          tokens: [B, T, cross_attn_dim]
          attn_mask: [B, T] (HD位置=1 其余=0; 若 UNet 支持可传, 否则可忽略)
        Notes:
          - Handles N=0/1/2 seamlessly
          - spread_radius: Spread HD influence to nearby r frames (0/1/2)
        """

        B, T, C, H, W = lr_frames.shape
        device = lr_frames.device

        # 1) LR 全序列编码
        lr_token = self._encode_frames_clip(lr_frames)  # [B, T, D]
        base_dtype = lr_token.dtype
        D = lr_token.size(-1)

        tokens = lr_token.clone()
        attn_mask = torch.zeros(B, T, device=device, dtype=lr_token.dtype)  # [B,T], HD位置=1 其余=0

        # 2) 若没有 HD 或 mask 全 False，直接返回 LR
        has_hd = (
            hd_frames is not None and
            indices_mask is not None and
            indices_mask.numel() > 0 and
            indices_mask.to(torch.bool).any().item()
        )
        if not has_hd:
            # 位置/来源嵌入 & dtype 统一
            pos_ids = torch.arange(T, device=device)[None, :].expand(B, T)
            tokens = tokens + 0.1 * self.pos_abs(pos_ids)
            src_ids = (attn_mask > 0).long()  # 全 0
            tokens = tokens + self.src_type(src_ids)
            tokens = tokens.to(self.pipe.unet.dtype)
            return tokens, attn_mask


        # 3) 标准化 fixed_indices -> [B, N]

        indices =  self.fixed_indices.view(1, self.N).expand(B, -1).to(self.device)  # [B,N]    
        N = indices.size(1)

        # 4) 取 LR/HD 在候选位置的 token
        ind_exp = indices.unsqueeze(-1).expand(B, N, D)                         # [B,N,D]
        lr_token_at_ind = lr_token.gather(1, ind_exp).to(base_dtype)        # [B,N,D]
        hd_token = self._encode_frames_clip(hd_frames).to(base_dtype)       # [B,N,D]

        # 5) 掩码选择：显式成对索引，避免跨样本混合
        mask = indices_mask.bool().to(device)
        b_idx, s_idx = mask.nonzero(as_tuple=True)      # [M], [M]
        true_ind = indices[b_idx, s_idx]                       # [M], 一个batch中, 所有被选中的hd frame 的序号
        lr_sel = lr_token_at_ind[b_idx, s_idx, :]       # [M, D], 一个batch中, 所有被选中的hd frame 同样位置的 lr frame的特征
        hd_sel = hd_token[b_idx, s_idx, :]              # [M, D], 一个batch中, 所有被选中的hd frame 的特征

        # 6) 门控融合（标量门；需要向量门时可替换）
        if self.use_adaptive_gate:
            feat = torch.cat([lr_sel, hd_sel, (hd_sel - lr_sel).abs()], dim=-1)  # [M, 3D]
            # print('Adaptive gate features:', feat.shape)
            g = torch.sigmoid(self.g_mlp(feat))                                  # [M,1] 或 [M,D]
        else:
            g = torch.sigmoid(self.hd_gate).view(1, 1).expand(hd_sel.size(0), 1) # [M,1]

        g = g.to(base_dtype)
        fused_sel = g * hd_sel + (1.0 - g) * lr_sel                               # [M, D]

        # 7) 写回中心位置（逐样本逐时刻，一一对应）
        tokens[b_idx, true_ind, :] = fused_sel
        attn_mask[b_idx, true_ind] = 1.0

        # 8) 邻域扩散 —— 按 batch 循环、逐个 video 处理（不跨样本）
        if spread_radius > 0:
            # 先把“中心写回后的 tokens”做快照，只用于读取原邻居，避免写后读
            tokens_base = tokens.clone()

            # 为了每个样本独立扩散：逐样本处理
            for i in range(B):
                # 取该样本的所有中心（b==i）
                sel_i = (b_idx == i)
                if not sel_i.any():
                    continue

                centers_i = true_ind[sel_i]          # [Mi]
                fused_i   = fused_sel[sel_i, :]   # [Mi, D]

                # 该样本的累加器/计数器，仅作用于第 i 个样本
                accum_i = torch.zeros((T, D), device=device, dtype=tokens.dtype)   # [T,D]
                count_i = torch.zeros((T, 1), device=device, dtype=tokens.dtype)   # [T,1]

                # 遍历 offset，对称扩散
                for offset in range(1, spread_radius + 1):
                    w = 1.0 / (offset + 1)

                    for dir in (-1, 1):
                        nb_pos = centers_i + dir * offset           # [Mi]
                        valid = (nb_pos >= 0) & (nb_pos < T)
                        if not valid.any():
                            continue

                        t_valid = nb_pos[valid]                     # [m]
                        f_valid = fused_i[valid, :]                 # [m,D]

                        # 从该样本的 tokens_base 读取原邻居 → 融合
                        orig_nb = tokens_base[i, t_valid, :]        # [m,D]
                        upd = w * f_valid + (1.0 - w) * orig_nb     # [m,D]

                        # 累加 + 计数（在样本 i 的局部累加器上）
                        # 用 scatter_add_ 按行聚合（可能同一 t 被多个中心命中）
                        accum_i.index_add_(0, t_valid, upd)
                        count_i.index_add_(0, t_valid, torch.ones((t_valid.size(0), 1),
                                                                device=device, dtype=tokens.dtype))

                        # 邻域 mask（样本 i 局部）：取最大
                        attn_mask[i, t_valid] = torch.maximum(
                            attn_mask[i, t_valid],
                            torch.full((t_valid.size(0),), 0.5 / offset, device=device, dtype=attn_mask.dtype)
                        )

                # 把该样本的邻域平均写回（只改写被命中的位置）
                hit_i = (count_i.squeeze(-1) > 0)                  # [T]
                if hit_i.any():
                    tokens[i, hit_i, :] = accum_i[hit_i, :] / count_i[hit_i, :].clamp_min(1e-8)

        # 9) 位置/来源嵌入 + dtype
        pos_ids = torch.arange(T, device=device)[None, :].expand(B, T)
        tokens = tokens + 0.1 * self.pos_abs(pos_ids)

        src_ids = (attn_mask > 0).long()
        tokens = tokens + self.src_type(src_ids)

        tokens = tokens.to(self.pipe.unet.dtype)

        return tokens, attn_mask

    def get_frame_latents(self, frames_tensor: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Encode video frames to latents with optional masking.
        frames_tensor: [B, T or N, 3, H, W]
        mask: [B, T or N] bool, True means "encode this frame"; if None, treat as all True.
        Returns: [B, T or N, C_latent, h, w]  (unselected positions are zero-filled)
        """
        assert frames_tensor.dim() == 5, f"Expected [B,TN,3,H,W], got {tuple(frames_tensor.shape)}"
        B, TN, C, H, W = frames_tensor.shape
        device = frames_tensor.device
        vae = self.pipe.vae
        vae_sf = getattr(self.pipe, "vae_scale_factor", 8)
        latent_ch = getattr(vae.config, "latent_channels", 4)
        h, w = H // vae_sf, W // vae_sf

        # mask 缺省 = 全 True
        if mask is None:
            mask = torch.ones(B, TN, dtype=torch.bool, device=device)
        else:
            assert mask.shape == (B, TN), f"mask must be [B,T], got {tuple(mask.shape)}"
            mask = mask.to(torch.bool).to(device)

        # 预分配输出（零占位，避免对未选中的帧做编码）
        # latents_out = torch.zeros(B * TN, latent_ch, h, w, device=self.device, dtype=vae.dtype)
        latents_out = torch.zeros(B * TN, latent_ch, h, w, device=self.device, dtype=self.amp_dtype)

        # 选中项的扁平索引
        if mask.any():
            # 展平成 [B*T]，取被选中的全局下标
            flat_mask = mask.reshape(-1)
            flat_idx = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)  # [M]
            # 取被选中的帧并编码
            frames_flat = frames_tensor.reshape(B * TN, C, H, W).to(vae.dtype)
            selected = frames_flat.index_select(0, flat_idx)  # [M, 3, H, W]

            latents_sel = []
            with torch.no_grad():
                chunk_size = max(int(getattr(self, "chunk_size", 8)), 1)
                for i in range(0, selected.size(0), chunk_size):
                    chunk = selected[i:i + chunk_size]
                    latent_chunk = self.pipe._encode_vae_image(
                        chunk,
                        device=self.device,
                        num_videos_per_prompt=1,
                        do_classifier_free_guidance=self.do_classifier_free_guidance,
                    )  # [m, C_latent, h, w]
                    latents_sel.append(latent_chunk)

            if len(latents_sel):
                latents_sel = torch.cat(latents_sel, dim=0)
                # print('latents_out.dtype:', latents_out.dtype) # torch.float16
                # print('latents_sel.dtype:', latents_sel.dtype) # torch.bfloat16
                latents_out[flat_idx] = latents_sel.to(latents_out.dtype)

        # 还原形状并做 scaling factor
        latents = latents_out.reshape(B, TN, latent_ch, h, w)
        # latents = latents * vae.config.scaling_factor
        return latents

    def decode_fn(self, latents, decode_chunk_size):
        ## to save memory, decode_chunk_size should be small
        video = self.pipe.decode_latents(latents, num_frames=latents.shape[1], decode_chunk_size=decode_chunk_size)  # Keep small chunk_size for finer memory control
        return video.permute(0, 2, 1, 3, 4).clamp(-1, 1)

    def compute_loss(self, lr_frames, hd_frames, indices_mask):
        '''
        Compute the loss for a batch of LR and HD frames.
        input:
            lr_frames: tensor (B, T, 3, H, W)
            hd_frames: tensor (B, N, 3, H, W)
            indices_mask: tensor (B, N) ( used to specify which HD frames are active)
            # sparse_indices: tensor (B, N) (indices of HD frames to use for conditioning), length is N.
        
        '''
        B, N, C_video, H_video, W_video = hd_frames.shape
        assert N == self.N, f"Expected N={self.N}, got {N}"

        sparse_indices = self.fixed_indices.view(1, self.N).expand(B, -1).to(self.device)  # [B,N]

        lr_frames = lr_frames.to(self.device, dtype=self.amp_dtype)
        hd_frames = hd_frames.to(self.device, dtype=self.amp_dtype)

        indices_mask = indices_mask.to(self.device, dtype=torch.bool)

        # 1) Get LR conditioning latents (separate)
        cond_lr_latents = self.get_frame_latents(lr_frames)
        B, T, C_latent, H_latent, W_latent = cond_lr_latents.shape
        latent_dtype = cond_lr_latents.dtype


        cond_hd_latents = self.get_frame_latents(hd_frames, indices_mask)  # [B, N, C_latent, H_latent, W_latent]
        # print('cond_hd_latents.dtype:', cond_hd_latents.dtype) # torch.bfloat16
        assert cond_hd_latents.dtype == latent_dtype, f"cond_hd_latents.dtype {cond_hd_latents.dtype} != cond_lr_latents.dtype {latent_dtype}"

        ## 把 N 个槽位的 HR latents “散射”到时间轴 [B,T,...]
        latent_ind_over_T = sparse_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(B, self.N, C_latent, H_latent, W_latent)
        hd_latent_over_T = torch.zeros(B, T, C_latent, H_latent, W_latent, device=lr_frames.device, dtype=latent_dtype)        
        # print('hd_latent_over_T.dtype:', hd_latent_over_T.dtype) # torch.bfloat16
        hd_latent_over_T.scatter_(1, latent_ind_over_T, cond_hd_latents)  # 未启用的槽位本身是全零


        ## 按 mask 构造时间权重图 w: [B,T,1,1,1]
        w = torch.zeros(B, T, 1, 1, 1, device=cond_hd_latents.device, dtype=cond_hd_latents.dtype)
        w.scatter_(
            1,
            sparse_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
            indices_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(cond_hd_latents.dtype),
        )



        ## 最终融合：只在 mask=True 的固定索引处，用 HR 覆盖 LR
        full_cond_latents = w * hd_latent_over_T + (1.0 - w) * cond_lr_latents  # [B,T,C_latent, H_latent, W_latent]


        # 2) Sample timestep
        t = torch.randint(0, self.pipe.scheduler.config.num_train_timesteps, (B,), dtype=torch.long).to(self.device)
        # Get added time IDs
        added_time_ids = self.get_added_time_ids(batch_size=B)

        # 3) add noise
        gt_noise = torch.randn_like(full_cond_latents, device=self.device)  # [B, T, C_latent, H_latent, W_latent]
        z_t = self.pipe.scheduler.add_noise(full_cond_latents, gt_noise, t)   ## [B, T, C_latent, H_latent, W_latent]

        # 4) Get HD embeddings with pos (separate)
        image_embeddings, _ = self.build_image_embeddings(lr_frames, hd_frames, sparse_indices)
        
        
        # 5) UNet 预测 velocity（保持 latent_dtype）
        ## Prepare UNet input: concat noisy target with LR conditioning
        latent_model_input = torch.cat([z_t, cond_lr_latents], dim=2)  # [1, T, 2*C, H_latent, W_latent]

        ##  Predict velocity
        vel_pred = self.pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=image_embeddings,
            added_time_ids=added_time_ids,
            return_dict=False
        )[0]   ### vel_pred: [B, T, C_latent, h, w], here is [B, 14, 4, 32, 32]
        # print('type of vel_pred:', vel_pred.dtype) #float16

        graph_zero = vel_pred.sum() * 0.0


        # 6) compute x0 from predicted velocity
        alpha_bar = self.pipe.scheduler.alphas_cumprod[t]      # [B]      float32       
        ## 扩成 [B,1,1,1,1] 便于广播到 [B,T,C,h,w]
        sqrt_ab  = alpha_bar.view(B, 1, 1, 1, 1).sqrt()
        sqrt_oma = (1.0 - alpha_bar).view(B, 1, 1, 1, 1).sqrt()
        ## 用 v→x0 公式直接求 pred_original
        #    x0_hat = sqrt(ab)*x_t - sqrt(1-ab)*vel_pred
        pred_original = (sqrt_ab * z_t.to(self.data_type) - sqrt_oma * vel_pred.to(self.data_type)).clamp(-1, 1)       # [B,T,C,h,w]
        pred_original = pred_original.to(latent_dtype)                           # ←★ 回到半精度


        # 7) 构造 v 的 GT（fp32）并计算 denoise loss（只在选中的槽位）
        ## 先取 \bar{alpha}_t，再构造 v_gt = sqrt(ab)*ε - sqrt(1-ab)*x0
        vel_gt = sqrt_ab * gt_noise.to(self.data_type) - sqrt_oma * full_cond_latents.to(self.data_type)                 # [B,T,C_latent, H_latent, W_latent]

        ##  还原全时序 HD 掩码：hd_mask_full[b, t]=True 表示该样本在 t 是 HD 槽且 active
        hd_mask_full = torch.zeros(B, T, dtype=torch.bool, device=self.device)
        bidx = torch.arange(B, device=self.device)
        for n in range(self.N):  # self.N=2
            t_idx = sparse_indices[:, n]                  # [B]
            act   = indices_mask[:, n].bool()            # [B]
            hd_mask_full[bidx[act], t_idx[act]] = True

        ##  按样本自适应权重(0.5~0.7) ：有 HD 的样本，令 HD 占比 = alpha；无 HD 的样本，整体弱化/关掉
        alpha = 0.65
        w = torch.zeros(B, T, dtype=torch.float32, device=self.device)
        k = hd_mask_full.sum(dim=1)  # [B]
        den_hd = k.clamp(min=1).unsqueeze(1)                 # [B,1]
        den_lr = (T - k).clamp(min=1).unsqueeze(1)           # [B,1]


        w_hd = (alpha / den_hd)                              # [B,1]
        w_lr = ((1.0 - alpha) / den_lr)                      # [B,1]

        w = torch.where(hd_mask_full, w_hd, w_lr)            # [B,T]

        ## k==0 样本：，整体弱化/关掉（几乎关闭 denoise）
        beta = 0.1
        no_hd = (k == 0)
        if no_hd.any():
            w[no_hd] = beta / T  # w[b].zero_() 或 w[b].fill_(0.02)

        w = w * T #让每样本权重和恒为 T，数值更稳
        max_w = 8.0  # 避免个别样本权重过大
        w = w.clamp(max=max_w)
        w = w.view(B, T, 1, 1, 1).to(self.data_type)

        ## 计算全帧加权 MSE（ fp32 计算）
        diff = (vel_pred.to(self.data_type) - vel_gt.to(self.data_type)) ** 2  # [B,T,C,H,W]
        L_denoise = (diff * w).mean().to(self.data_type)



        if self.use_latent_warp:
            # 8) Decode to pixel space
            with torch.no_grad():            
                gen_video = self.decode_fn(pred_original, decode_chunk_size = 14)  # 默认decode_chunk_size=14, [B, C_video, T, H_video, W_video]

            pred_original = pred_original.to(self.data_type)
            # 9) 计算光流损失
            L_temp = latent_warp_flow_loss(
            raft_model         = self.raft,
            pred_x0_latents_f32 = pred_original,   # [B,T,C_lat,hz,wz]
            frames_pixels           = lr_frames,     # [B,T,3,H,W], 任意精度输入，这里内部会转 [0,1]
            step                = 2,
            raft_scale          = 0.5
            )

            # 10) 计算重建loss
            if indices_mask.any().item():
                pred_latent_sel = pred_original.gather(1, latent_ind_over_T) # [B,N,C_lat,hz,wz]
                pred_flat = pred_latent_sel[indices_mask]  # [M,C,h,w]
                gt_flat = cond_hd_latents[indices_mask]  # [M,C,h,w]
                L_rec = F.mse_loss(pred_flat, gt_flat.to(self.data_type), reduction='mean')
            else:
                L_rec = graph_zero

        else:  
            # 8) Decode to pixel space 
            # Use checkpoint to decode with gradients but lower memory
            gen_video = checkpoint(self.decode_fn, pred_original, decode_chunk_size=2, use_reentrant=False)
            # print(f'gen_video shape: {gen_video.shape}')


            # 9) 像素监督 rec loss（有 HD 才算）
            video_ind_over_T = sparse_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(B, self.N, C_video, H_video, W_video)
            gen_hd_video_sel = gen_video.gather(1, video_ind_over_T) # [B, N, C, H, W]
            
            if indices_mask.any().item():
                # 为数值稳定，可以用 float32 算 loss，再把标量转回 self.data_type
                gen_hd_video_flat = gen_hd_video_sel[indices_mask]  # [M, C, H_video, W_video]
                gt_hd_video_flat  = hd_frames[indices_mask]    # [M, C, H_video, W_video]
                L_rec = F.mse_loss(gen_hd_video_flat.to(self.data_type), gt_hd_video_flat.to(self.data_type), reduction='mean')
                # Perceptual loss (average over frames)
                # M = gen_hd_video_flat.shape[0]
            
                # L_perc = self.lpips(gen_hd_video_flat, gt_hd_video_flat).to(self.data_type)
                # L_perc /= M            
                
            else:
                L_rec = graph_zero
                # L_perc = torch.tensor(0.0, device=self.device, dtype=self.data_type)  


            # 10) 光流loss（fp32）
            lr_norm = (lr_frames.clamp(-1, 1) + 1) / 2
            gen_norm = (gen_video.clamp(-1, 1) + 1) / 2        
            L_temp = compute_temporal_loss(gen_norm, lr_norm, step = 2, scale = 0.5)


        # print('type of L_denoise:', L_denoise.dtype)
        # print('type of L_fid:', L_fid.dtype)
        # print('type of L_lr_temp:', L_lr_temp.dtype)
        # print('type of L_hd_temp:', L_hd_temp.dtype)

        # 11)  Total loss
        loss = L_denoise + 0.5 * L_rec + 0.01 * L_temp 
        # loss = loss.to(self.data_type)  # Ensure same dtype as UNet
        print(f"Loss: L_denoise={L_denoise.item():.4f}, L_rec={L_rec.item():.4f}, L_temp={L_temp.item():.4f}")
        return loss

    def train(self, dataset, epochs=100, debug = False, grad_accum=1):
        use_scaler = (self.amp_dtype == torch.float16)
        scaler = GradScaler("cuda", enabled=use_scaler)



        for epoch in range(epochs):
            total_loss = 0.0
            self.optimizer.zero_grad(set_to_none=True)

            for i, batch in enumerate(dataset):
                lr_frames, hd_frames, mask = batch

                with autocast("cuda", dtype=self.amp_dtype, enabled=use_scaler):
                    loss = self.compute_loss(lr_frames, hd_frames, mask) / grad_accum

                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (i + 1) % grad_accum == 0:
                    if use_scaler:
                        scaler.step(self.optimizer); scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                total_loss += loss.item() * grad_accum
                if debug and i > 5:
                    break

            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}\n")
            if debug and epoch > 2:
                break
    print("Training complete.")



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = DummyFewShotDataset()
    trainer = FewShotVideoSRTrainer(device=device)
    trainer.train(dataset, epochs=10, debug=True)
