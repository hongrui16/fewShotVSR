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



class FewShotVideoSRTrainer:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.motion_bucket_id = 127  # As in SVD, motion_bucket_id is 127
        self.noise_aug_strength = 0.02
        self.num_frames = 14  # For SVD, max frames is 14
        self.embed_dim = 768  # CLIP embed dimension
        # Load SVD pipeline
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            torch_dtype=torch.float16
        ).to(self.device)

        self.unet_cross_attention_dim = self.pipe.unet.config.cross_attention_dim
        self.hd_gate = nn.Parameter(torch.tensor(-1.0, device= self.device))  # Init low (~0.27) for early stability
        self.g_mlp = nn.Sequential(
            nn.Linear(3 * self.embed_dim, 128),  # For adaptive gate: input [LR, HD, |diff|] = 3D
            nn.ReLU(),
            nn.Linear(128, 1)  # Output scalar per-frame g
        ).to(self.device)
        self.pos_abs = nn.Embedding(self.num_frames, self.embed_dim).to(self.device)  # Learnable positional encoding
        self.src_type = nn.Embedding(2, self.embed_dim).to(self.device)  # 0: LR, 1: HD
        self.cross_proj = nn.Linear(self.embed_dim, self.unet_cross_attention_dim).to(self.device)  # Projection if needed
        self.cond_ln = nn.LayerNorm(self.unet_cross_attention_dim).to(self.device)
        self.T_max = self.num_frames  # 14

        # Freeze VAE
        for param in self.pipe.vae.parameters():
            param.requires_grad = False
        print("VAE frozen.")

        # Apply LoRA to U-Net
        lora_config = LoraConfig(
            r=8,
            lora_alpha=4,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.1
        )
        self.pipe.unet = get_peft_model(self.pipe.unet, lora_config)
        print("LoRA applied.")

        self.chunk_size = 8
       
        # LPIPS
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(self.device)

        # RAFT for temporal loss
        self.raft = raft_small(pretrained=True).eval().to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.pipe.unet.parameters()), lr=1e-5)

        # Scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

    

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
    

    def compute_temporal_loss(self, generated_frames, gt_frames):
        """
        Compute temporal consistency loss based on optical flow.
        
        generated_frames: [B, 3, T, H, W] - generated video
        gt_frames:        [B, 3, T, H, W] - ground truth reference video (LR or HD)
        """
        B, C, T, H, W = generated_frames.shape

        # 帧数不足 2 无法计算光流
        if T <= 1:
            return torch.tensor(0.0, device=self.device)

        generated_frames = generated_frames.permute(0, 2, 1, 3, 4)  # [B, T, 3, H, W]
        gt_frames = gt_frames.permute(0, 2, 1, 3, 4)  # [B, T, 3, H, W]

        L_temp = 0.0
        count = 0

        for b in range(B):
            for i in range(T - 1):
                gen1 = generated_frames[b, i].unsqueeze(0)
                gen2 = generated_frames[b, i + 1].unsqueeze(0)
                gt1 = gt_frames[b, i].unsqueeze(0)
                gt2 = gt_frames[b, i + 1].unsqueeze(0)

                with torch.no_grad():
                    flow_gen = self.raft(gen1, gen2)[-1]  # [1, 2, H, W]
                    flow_gt = self.raft(gt1, gt2)[-1]

                L_temp += nn.MSELoss()(flow_gen, flow_gt)
                count += 1

        return L_temp / count if count > 0 else torch.tensor(0.0, device=self.device)


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


    def get_hd_embeddings(self, hd_frames, sparse_indices):
        """
        hd_frames: tensor (B, 3, N, H, W) HD视频帧
        sparse_indices: tensor (B, N) 对应帧索引（顺序对应 hd_frames）
        Returns: tensor (B, N, embed_dim)
        """
        B, C, N, H, W = hd_frames.shape
        hd_frames = hd_frames.permute(0, 2, 1, 3, 4)  # [B, N, 3, H, W]
        # 展平 batch 和时间维度，送入 CLIP image encoder
        frames_flat = hd_frames.reshape(B * N, C, H, W) # [BN, 3, H, W]
        # 提取图像特征
        embeds = self.pipe._encode_image(frames_flat, self.device, 1, False) # [BN embed_dim]
        embeds = embeds.view(B, N, -1) # [B, N, embed_dim]
        # 生成位置编码
        pe = self.positional_encoding(sparse_indices, max_pos=self.num_frames, d_model=self.embed_dim) # [B, N, embed_dim]
        return embeds + pe

    @torch.no_grad()
    def _encode_frames_clip(self, frames):
        """
        Helper to encode frames with CLIP: [B, C, T, H, W] -> [B, T, D]
        Handles chunking if needed.
        """
        B, C, T, H, W = frames.shape
        flat = frames.permute(0,2,1,3,4).reshape(B*T, C, H, W)

        embeds = self.pipe._encode_image(flat, self.device, 1, False).reshape(B, T, -1)
        return embeds

    def build_image_embeddings(self, lr_frames, hd_frames=None, sparse_indices=None,
                               use_adaptive_gate=False, spread_radius=0):
        """
        Returns:
          tokens: [B, T, cross_attn_dim]
          attn_mask: [B, T] (HD位置=1 其余=0; 若 UNet 支持可传, 否则可忽略)
        Notes:
          - Handles N=0/1/2 seamlessly
          - spread_radius: Spread HD influence to nearby r frames (0/1/2)
        """
        B, C, T, H, W = lr_frames.shape
        assert T <= self.T_max
        device = lr_frames.device

        # 1) LR full sequence base
        lr_tok = self._encode_frames_clip(lr_frames)  # [B, T, D]
        tokens = lr_tok.clone()
        attn_mask = torch.zeros(B, T, device=device)

        has_hd = (hd_frames is not None) and (sparse_indices is not None) and (sparse_indices.numel() > 0)
        if has_hd:
            B2, C2, N, _, _ = hd_frames.shape
            assert B2 == B and C2 == 3 and N <= T
            hd_tok = self._encode_frames_clip(hd_frames)  # [B, N, D]

            idx = sparse_indices.long()  # [B, N]
            D = lr_tok.size(-1)
            idx_exp = idx.unsqueeze(-1).expand(B, N, D)
            lr_at_idx = lr_tok.gather(1, idx_exp)  # [B, N, D]

            # Gated fusion
            if use_adaptive_gate:
                feat = torch.cat([lr_at_idx, hd_tok, (hd_tok - lr_at_idx).abs()], dim=-1)  # [B, N, 3D]
                g = torch.sigmoid(self.g_mlp(feat))  # [B, N, 1]
            else:
                g = torch.sigmoid(self.hd_gate).view(1, 1, 1)  # Scalar [1,1,1]

            fused = g * hd_tok + (1 - g) * lr_at_idx  # [B, N, D]
            tokens.scatter_(1, idx_exp, fused)  # Write back
            attn_mask.scatter_(1, idx, torch.ones_like(idx, dtype=attn_mask.dtype))


            # Optional: Neighborhood spread with accumulated weights
            if spread_radius > 0:
                # Loop over batch and HD positions to handle spreads individually
                # Note: This handles overlaps by sequential updates; for small N/r, accumulation is minimal.
                # If overlaps are frequent, consider collecting all contributions and averaging (more complex).
                for i in range(B):
                    for j in range(N):
                        center = idx[i, j].item()
                        center_fused = fused[i+1, j:j+1]  # [1, D] (keep dim for scatter)
                        for offset in range(1, spread_radius + 1):
                            w = 1.0 / (offset + 1)  # Decaying weight
                            for dir in (-1, 1):
                                nb_pos = center + dir * offset
                                if 0 <= nb_pos < T:
                                    nb_idx = torch.full((1, 1, tokens.size(-1)), nb_pos, device=device, dtype=torch.long)           # [1,1,D]
                                    orig_nb = tokens[i:i+1].gather(1, nb_idx)  # [1, 1, D]
                                    updated_nb = w * center_fused + (1 - w) * orig_nb  # [1, 1, D]
                                    tokens[i:i+1].scatter_(1, nb_idx, updated_nb)
                                    # Gradual mask: decay from 1.0 at center
                                    attn_mask[i, nb_pos] = torch.maximum(attn_mask[i, nb_pos], torch.tensor(0.5/offset, device=device))


        # 2) Add positional + source type embeddings
        pos_ids = torch.arange(T, device=device)[None, :].expand(B, T)
        tokens = tokens + 0.1 * self.pos_abs(pos_ids)  # Scaled learnable PE
        src_ids = (attn_mask > 0).long()  # 0: LR, 1: HD/fused
        tokens = tokens + self.src_type(src_ids)

        # 3) Project to cross-attn dim and LN
        tokens = self.cond_ln(self.cross_proj(tokens))  # [B, T, embed_dim]
        tokens = tokens.to(self.pipe.unet.dtype)

        return tokens, attn_mask

    def get_frame_latents(self, frames_tensor: torch.Tensor):
        """
        Encode video frames to latents using _encode_vae_image, handling multi-frame.
        frames_tensor: [B, 3, T, H, W]
        Returns: [B, C_latent, T, h, w]
        """
        B, C, T, H, W = frames_tensor.shape
        frames_flat = frames_tensor.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # [B*T, 3, H, W]
        
        latents = []
        with torch.no_grad():
            chunk_size = self.chunk_size  # 调整以防OOM，小于T即可
            for i in range(0, B * T, chunk_size):
                chunk = frames_flat[i:i + chunk_size]
                # 调用管道的_encode_vae_image，设置指导为False，num=1
                latent_chunk = self.pipe._encode_vae_image(
                    chunk,
                    device=self.device,
                    num_videos_per_prompt=1,
                    do_classifier_free_guidance=False
                )  # [chunk_size, C_latent, h, w]
                latents.append(latent_chunk)
        
        latents = torch.cat(latents, dim=0)  # [B*T, C_latent, h, w]
        latents = latents.reshape(B, T, *latents.shape[1:]).permute(0, 2, 1, 3, 4)  # [B, C_latent, T, h, w]
        
        # SVD latent通常需乘scaling factor（从VAE config获取）
        latents = latents * self.pipe.vae.config.scaling_factor
        
        return latents
    
    def compute_loss(self, lr_frames, hd_frames, sparse_indices):
        '''
        Compute the loss for a batch of LR and HD frames.
        
        lr_frames: tensor (B, 3, T, H, W)
        hd_frames: tensor (B, 3, N, H, W)
        sparse_indices: tensor (B, N) (indices of HD frames to use for conditioning), length is N.
        
        '''


        # Get LR conditioning latents (separate)
        cond_lr_latents = self.get_frame_latents(lr_frames)

        hr_latents = self.get_frame_latents(hd_frames)  # 

        B, C, T, H, W = cond_lr_latents.shape

        full_hr_latents = cond_lr_latents.clone()  # Start with LR everywhere

        indices = sparse_indices.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand(B, C, N, H, W).long() 

        full_hr_latents.scatter_(dim=2, index=indices, src=hr_latents)  # Place HR at sparse indices

        
        # Sample timestep
        t = torch.randint(0, self.pipe.scheduler.config.num_train_timesteps, (B,)).to(self.device)
        
        ### add noise
        gt_noise = torch.randn_like(full_hr_latents)
        z_t = self.pipe.scheduler.add_noise(full_hr_latents, gt_noise, t)

        
        # Get HD embeddings with pos (separate)
        image_embeddings = self.build_image_embeddings(lr_frames, hd_frames, sparse_indices)
        
        # Get added time IDs
        added_time_ids = self.get_added_time_ids(batch_size=B)
        
        # Prepare UNet input: concat noisy target with LR conditioning
        latent_model_input = torch.cat([z_t, cond_lr_latents], dim=1)  # [1, 2*C, T, h, w]
        
        # Predict noise
        predicted_noise = self.pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=image_embeddings,
            added_time_ids=added_time_ids,
            return_dict=False
        )[0]
        
        # Denoising loss

        pre_noise_selected = predicted_noise.gather(2, indices)
        gt_noise_selected = gt_noise.gather(2, indices)
        L_denoise = nn.MSELoss()(pre_noise_selected, gt_noise_selected)

        # Get predicted original sample for additional losses
        pred_original = self.pipe.scheduler.step(predicted_noise, t, z_t).pred_original_sample # [B, C, T, h, w]
        pred_original = pred_original.clamp(-1, 1)  # [B, C, T, h, w]

        
        # Decode to pixel space
        generated_video = self.pipe.decode_latents(pred_original, num_frames=pred_original.shape[2])  # 默认decode_chunk_size=14

        generated_video_selected = generated_video.gather(2, indices)
        # Fidelity loss (L1 on full video)
        L_fid = nn.L1Loss()(generated_video_selected, hd_frames)  # Adjust dims if needed
        
        # Perceptual loss (average over frames)

        L_perc = 0.0
        N = hd_frames.shape[2]
        for i in range(N):
            L_perc += self.lpips(generated_video_selected[:, :, i, :, :], hd_frames[:, :, i, :, :])
        L_perc /= N

        # Temporal loss (normalize frames to [0,1] for RAFT)
        gen_norm = (generated_video.clamp(-1, 1) + 1) / 2
        lr_norm = (lr_frames.clamp(-1, 1) + 1) / 2
        hd_selected_norm = (generated_video_selected.clamp(-1, 1) + 1) / 2
        hd_norm = (hd_frames.clamp(-1, 1) + 1) / 2
        L_lr_temp = self.compute_temporal_loss(gen_norm, lr_norm)
        L_hd_temp = self.compute_temporal_loss(hd_selected_norm, hd_norm)
        
        # Total loss
        return L_denoise + 0.5 * L_fid + 0.5 * L_perc + 0.2 * L_lr_temp + 0.1 * L_hd_temp

    def train(self, dataset, epochs=100):
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in dataset:
                # Assume batch = (lr_frames, hd_frames, sparse_indices)
                lr_frames, hd_frames, sparse_indices = batch
                loss = self.compute_loss(lr_frames, hd_frames, sparse_indices)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")
        print("Training complete.")


# Dummy dataset
hr_lr_dataset = [
    (np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
     np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    for _ in range(3)
]

trainer = FewShotVideoSRTrainer(device="cuda")
trainer.train(hr_lr_dataset, epochs=10)
