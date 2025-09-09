import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import StableVideoDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import DDPMScheduler

from peft import LoraConfig, get_peft_model
from peft import get_peft_model_state_dict
from peft import set_peft_model_state_dict
from contextlib import nullcontext
from peft import PeftModel
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPProcessor
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from torchvision.models.optical_flow import raft_small
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.utils import DistributedDataParallelKwargs
import logging
from transformers import get_cosine_schedule_with_warmup

from datetime import datetime
import glob
import importlib
import shutil
import math
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Dict, Any
import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys, os
# sys.path.append(os.path.dirname(__file__))  # 把当前目录加入路径

from torch.amp import GradScaler, autocast




from config.config import cfg
from config.config import parse_args

# from dataloader.dataset.dummy_dataset import DummyFewShotDataset
from dataloader.build_dataloader import build_dataloader
from utils.vis import visualize_loss
from utils.loss_func import compute_temporal_loss, latent_warp_flow_loss

class SVDForSR(StableVideoDiffusionPipeline):
    def __call__(
        self,
        *args,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 1.0,
        num_videos_per_prompt: int = 1,
        **kwargs,
    ):
        # 强制覆盖默认参数
        return super().__call__(
            *args,
            min_guidance_scale=min_guidance_scale,
            max_guidance_scale=max_guidance_scale,
            num_videos_per_prompt=num_videos_per_prompt,
            **kwargs,
        )

class SRModulesUnfrozen(nn.Module):
    def __init__(self, unet, g_mlp, pos_abs, src_type):
        super().__init__()
        self.unet = unet
        self.g_mlp = g_mlp
        self.pos_abs = pos_abs
        self.src_type = src_type
        
        D = unet.config.cross_attention_dim
        self.cond_pre_ln  = nn.LayerNorm(D, eps=1e-6)
        self.cond_adapter = nn.Sequential(
            nn.Linear(D, 2*D), nn.GELU(), nn.Linear(2*D, D)
        )
        # 残差零初始化：起步“无影响”
        nn.init.zeros_(self.cond_adapter[-1].weight)
        nn.init.zeros_(self.cond_adapter[-1].bias)
        


class FewShotVideoSRWorker:
    """
    Few-shot SVD-based SR with sparse HR conditioning (Accelerate-ready).


    Requirements satisfied:
    - No bfloat16 (we use fp16 AMP via Accelerator).
    - Train/Test modes.
    - Distributed & mixed precision through HuggingFace Accelerate.
    - UNet updated via LoRA (VAE frozen).
    - Save/Load checkpoints including LoRA adapters.
    """
    def __init__(
    self,
    args = None,
    **kwargs
    ) -> None:
        
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


        self.motion_bucket_id = 127  # As in SVD, motion_bucket_id is 127
        self.noise_aug_strength = 0.02
        self.num_frames = 14  # For SVD, max frames is 14
        self.embed_dim = 768  # CLIP embed dimension
        
        self.lamda_denoise = 1.0
        self.lamda_rec = 0.3
        self.lamda_tempor = 8.0
        self.use_adaptive_gate = True

        self.use_latent_warp = cfg.use_latent_warp
        self.finetune = cfg.finetune
        self.resume_path = cfg.resume_path
        self.num_train_epochs = cfg.num_train_epochs
        self.train_batch_size = cfg.train_batch_size
        
        
        self.mode = args.mode
        self.debug = args.debug
        enable_xformers_memory_efficient_attention = args.enable_xformers_memory_efficient_attention
        
        mixed_precision = args.mixed_precision
        gradient_accumulation_steps = args.gradient_accumulation_steps
        

        current_timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        millisecond = datetime.now().strftime('%f')
        current_timestamp = current_timestamp + '-' + millisecond
        job_id = os.getenv("SLURM_JOB_ID")
        if self.mode == 'train':
            self.logging_dir = os.path.join(cfg.log_dir, cfg.dataset_name, current_timestamp + '_' + job_id)
            # self.output_vis_dir = os.path.join(self.logging_dir, "vis")   
            if self.debug:
                self.output_ckpts_dir = self.logging_dir
            else:
                self.output_ckpts_dir = os.path.join(cfg.weight_output_dir, cfg.dataset_name, current_timestamp + '_' + job_id)
            
        elif self.mode == 'test':
            weight_dir = os.path.dirname(cfg.resume_path)
            parent_log_dir = weight_dir.replace(cfg.weight_output_dir, cfg.log_dir) 
            self.logging_dir = os.path.join(parent_log_dir, f'{current_timestamp}') 
            # self.output_ckpts_dir = weight_dir

        

        accelerator_project_config = ProjectConfiguration(project_dir=self.logging_dir, logging_dir=self.logging_dir)
        self.accelerator = Accelerator(
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    mixed_precision=mixed_precision,
                    project_config=accelerator_project_config,
                    kwargs_handlers=[
                    DistributedDataParallelKwargs(find_unused_parameters=True)]
                )
        self.accelerator.wait_for_everyone()
        self.device = self.accelerator.device
        
        if self.accelerator.is_main_process:
            os.makedirs(self.logging_dir, exist_ok=True)
            if self.mode == 'train':
                os.makedirs(self.output_ckpts_dir, exist_ok=True)

        log_path = os.path.join(self.logging_dir, "info.log")
        if self.accelerator.is_main_process:
            shutil.copyfile("config/config.py", os.path.join(self.logging_dir, f"config_{job_id}.py"))
            shutil.copyfile("fewShotVSR.py", os.path.join(self.logging_dir, f"fewShotVSR_{job_id}.py"))

        handlers = []
        if self.accelerator.is_main_process:
            # 主进程记录到文件和控制台
            handlers = [logging.FileHandler(log_path), logging.StreamHandler()]
            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
                level=logging.INFO,
                handlers=handlers,
                force=True  # 覆盖已有配置，避免重复
            )
        
        else:
            # 非主进程装一个 NullHandler，不输出
            logging.basicConfig(handlers=[logging.NullHandler()], force=True)

        self.logger = get_logger(__name__)

        self.logger.info(f"logging_dir:\n {self.logging_dir}")
        self.logger.info(f"output_ckpts_dir:\n {self.output_ckpts_dir}")
        
        # Load SVD pipeline
        self.pipe = SVDForSR.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            torch_dtype=torch.float16
        )

        # Dtypes
        self.unet_dtype = self.pipe.unet.dtype # fp16 under autocast
        self.data_type = torch.float32 # compute loss in fp32 then cast back
        self.amp_dtype = torch.float16 # strictly not bf16
        self.logger.info(f"unet_dtype: {self.unet_dtype}, data_type: {self.data_type}, amp_dtype: {self.amp_dtype}", main_process_only=self.accelerator.is_main_process)


        # 推理用（pipe.scheduler）：DPM-Solver，确保 v_pred
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler.register_to_config(prediction_type="v_prediction")
        self.logger.info(f'initializing a inference scheduler, {self.pipe.scheduler}', main_process_only=self.accelerator.is_main_process)

        # 训练用（独立的噪声调度器）
        self.noise_scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)
        self.noise_scheduler.register_to_config(prediction_type="v_prediction")
        self.logger.info(f'initialized a training scheduler, {self.noise_scheduler}', main_process_only=self.accelerator.is_main_process)


        # if not hasattr(self.pipe, "do_classifier_free_guidance"):
        #     self.do_classifier_free_guidance = False
        # else:
        #     self.do_classifier_free_guidance = self.pipe.do_classifier_free_guidance
        self.do_classifier_free_guidance = False  # No CFG for few-shot VSR
        self.logger.info(f'self.do_classifier_free_guidance: {self.do_classifier_free_guidance}', main_process_only=self.accelerator.is_main_process)

        
        # ===== Build gating / embeddings =====
        self.unet_cross_attention_dim = self.pipe.unet.config.cross_attention_dim
        self.logger.info(f'self.unet_cross_attention_dim: {self.unet_cross_attention_dim}', main_process_only=self.accelerator.is_main_process)
        
        
        self.g_mlp = nn.Sequential(
            nn.Linear(3 * self.unet_cross_attention_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
            )
        
        self.pos_abs = nn.Embedding(self.num_frames, self.unet_cross_attention_dim)
        self.src_type = nn.Embedding(2, self.unet_cross_attention_dim) # 0: LR, 1: HD
        self.logger.info(f'pos_abs: {self.pos_abs.weight.shape}', main_process_only=self.accelerator.is_main_process)
        self.logger.info(f'src_type: {self.src_type.weight.shape}', main_process_only=self.accelerator.is_main_process)

        # ===== Apply LoRA to UNet (trainable adapters; base UNet weights frozen by PEFT) =====
        # 最小增强版：
        attn_keys = ["to_q","to_k","to_v","to_out.0"]
        ffn_keys  = ["ff.net.0.proj","ff.net.2"]         # 视模型实际命名微调
        time_keys = ["time_embedding.linear_1","time_embedding.linear_2"]
        targets = attn_keys + ffn_keys + time_keys

        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=4,
            target_modules=targets,
            lora_dropout=0.1,
            )
        self.pipe.unet = get_peft_model(self.pipe.unet, lora_cfg)
        
        if self.accelerator.is_main_process:
            total = sum(p.numel() for p in self.pipe.unet.parameters())
            trainable = sum(p.numel() for p in self.pipe.unet.parameters() if p.requires_grad)
            self.logger.info(f"UNet params: {total/1e6:.2f}M, trainable (LoRA) ~ {trainable/1e6:.2f}M")



        self.unfrozenModules = SRModulesUnfrozen(
            unet=self.pipe.unet,
            g_mlp=self.g_mlp,
            pos_abs=self.pos_abs,
            src_type=self.src_type
        )
        
        
        self.pipe.unet = self.unfrozenModules.unet
        self.g_mlp     = self.unfrozenModules.g_mlp
        self.pos_abs   = self.unfrozenModules.pos_abs
        self.src_type  = self.unfrozenModules.src_type
        self.cond_pre_ln  = self.unfrozenModules.cond_pre_ln
        self.cond_adapter = self.unfrozenModules.cond_adapter

        # LPIPS
        # self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').eval()


        # Gradient checkpointing before prepare
        if self.mode == "train":
            try:
                self.pipe.unet.enable_gradient_checkpointing()
                self.logger.info("Enabled gradient checkpointing.", main_process_only=self.accelerator.is_main_process)
            except Exception:
                pass


        # xFormers (optional)
        if enable_xformers_memory_efficient_attention:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                self.logger.info("Enabled xFormers.", main_process_only=self.accelerator.is_main_process)
            except Exception:
                pass


        self.T_max = self.num_frames  # 14

        mid = (self.T_max - 1) // 2
        self.fixed_indices = torch.tensor([0, mid], dtype=torch.long, device=self.device)

        self.N = 2  # 槽位数（few-shot 数）
        
        ### dataset and dataloader initalization

        if self.debug:
            batch_size = 2
        else:
            batch_size = self.train_batch_size
        self.train_loader, self.train_dataset = build_dataloader(split='train', batch_size=batch_size, num_workers=4, logger=self.logger, device=self.device,
                                                                dataset_name=cfg.dataset_name
                                                                 )
        self.test_loader, self.test_dataset = build_dataloader(split='test', batch_size=batch_size, num_workers=4, logger=self.logger, device=self.device,
                                                               dataset_name=cfg.dataset_name
                                                               )

        self.height = self.train_dataset.height
        self.width = self.train_dataset.width
        self.logger.info(f"Dataset: {cfg.dataset_name}, Train samples: {len(self.train_dataset)}, Test samples: {len(self.test_dataset)}", main_process_only=self.accelerator.is_main_process)

        # ===== Optimizer (train only; only LoRA + small heads trainable) =====
        self.optimizer = None
        if self.mode == "train":
            trainable_params = []
            for m in [self.pipe.unet, self.g_mlp, self.pos_abs, self.src_type, self.cond_pre_ln, self.cond_adapter]:
                for p in m.parameters():
                    if p.requires_grad:
                        trainable_params.append(p)

            
            self.logger.info('initializing optimizer', main_process_only=self.accelerator.is_main_process)
            self.optimizer = optim.AdamW(
                                trainable_params,
                                lr=cfg.optimizer_dict['lr'],                 # 学习率
                                betas=cfg.optimizer_dict['adam_betas'],    # 常见设置
                                eps=cfg.optimizer_dict['adam_epsilon'],              # 数值稳定性
                                weight_decay=cfg.optimizer_dict['weight_decay']      # 官方推荐 0.01
                            )
            

            # ===== Prepare with Accelerator =====
        if self.mode == "train":
            (self.pipe.unet, self.g_mlp, self.pos_abs, self.src_type,
            self.cond_pre_ln, self.cond_adapter,
            self.optimizer, self.train_loader) = self.accelerator.prepare(
                self.pipe.unet, self.g_mlp, self.pos_abs, self.src_type,
                self.cond_pre_ln, self.cond_adapter,
                self.optimizer, self.train_loader
            )
        else:
            (self.pipe.unet, self.g_mlp, self.pos_abs, self.src_type,
            self.cond_pre_ln, self.cond_adapter,
            self.test_loader) = self.accelerator.prepare(
                self.pipe.unet, self.g_mlp, self.pos_abs, self.src_type,
                self.cond_pre_ln, self.cond_adapter,
                self.test_loader
            )

        if self.mode == "train":
            steps_per_epoch   = math.ceil(len(self.train_loader) / gradient_accumulation_steps)  # 这里的 len 是“每进程”的
            num_training_steps= steps_per_epoch * self.num_train_epochs
            num_warmup_steps  = int(num_training_steps * cfg.warmup_ratio)

            self.logger.info(
                f"Preparing for training: {self.num_train_epochs} epochs, "
                f"{steps_per_epoch} steps/epoch, {num_training_steps} total steps "
                f"({num_warmup_steps} warmup)",
                main_process_only=self.accelerator.is_main_process
            )

            self.logger.info('initializing lr scheduler', main_process_only=self.accelerator.is_main_process)
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )


        
        # Move non-prepared modules
        self.logger.info('moving non-prepared modules to device', main_process_only=self.accelerator.is_main_process)
        self.pipe.vae.eval().requires_grad_(False).to(self.device)
        self.logger.info('moving VAE to device', main_process_only=self.accelerator.is_main_process)
        
        self.pipe.image_encoder.eval().requires_grad_(False).to(self.device)
        self.logger.info('moving CLIP image encoder to device', main_process_only=self.accelerator.is_main_process)


        # RAFT for temporal loss
        self.raft = raft_small(pretrained=True)
        self.logger.info("RAFT model loaded.", main_process_only=self.accelerator.is_main_process)

        self.raft.eval()
        self.logger.info('moving RAFT to device', main_process_only=self.accelerator.is_main_process)
        for p in self.raft.parameters(): p.requires_grad = False
        self.raft.to(self.device)

    


        self.start_epoch = 0

        self.logger.info("Initialization complete.\n", main_process_only=self.accelerator.is_main_process)

    def save_checkpoint(self, epoch, is_best_epoch=False):
        fname = "best_checkpoint.pt" if is_best_epoch else "checkpoint.pt"
        path = os.path.join(self.output_ckpts_dir, fname)

        if self.accelerator.is_main_process:
            # 统一从“真实 forward 的模块”解包
            unet_raw    = self.accelerator.unwrap_model(self.pipe.unet)
            g_mlp_raw   = self.accelerator.unwrap_model(self.g_mlp)
            pos_abs_raw = self.accelerator.unwrap_model(self.pos_abs)
            src_type_raw= self.accelerator.unwrap_model(self.src_type)
            pre_ln_raw  = self.accelerator.unwrap_model(self.cond_pre_ln)
            adapter_raw = self.accelerator.unwrap_model(self.cond_adapter)

            # 1) 保存 UNet：优先保存 LoRA，若不是 PeftModel 再保存整网
            state = {}
            if isinstance(unet_raw, PeftModel) or hasattr(unet_raw, "peft_config"):
                state["unet_lora"] = get_peft_model_state_dict(unet_raw)
                self.logger.info("Saving LoRA weights.", main_process_only=True)
            else:
                state["unet"] = unet_raw.state_dict()
                self.logger.info("Saving full UNet weights.", main_process_only=True)

            # 2) 保存自定义头/嵌入/适配层
            state.update({
                "g_mlp":        g_mlp_raw.state_dict(),
                "pos_abs":      pos_abs_raw.state_dict(),
                "src_type":     src_type_raw.state_dict(),
                "cond_pre_ln":  pre_ln_raw.state_dict(),
                "cond_adapter": adapter_raw.state_dict(),
                "epoch":        epoch,
            })

            # 3) 训练场景下保存优化器与调度器
            if self.mode == "train":
                state["optimizer"] = self.optimizer.state_dict()
                state["scheduler"] = self.scheduler.state_dict()

            # 4) 只主进程写盘
            if os.path.exists(path):
                os.remove(path)
            self.accelerator.save(state, path)

        self.accelerator.wait_for_everyone()
        self.logger.info(f"Saved checkpoint to {path}", main_process_only=self.accelerator.is_main_process)


    def load_checkpoint(self):
        if self.resume_path is None or not os.path.isfile(self.resume_path):
            self.logger.warning(f"Resume path is invalid: {self.resume_path}", main_process_only=self.accelerator.is_main_process)
            return

        # 只让主进程读盘，其它进程等
        with self.accelerator.main_process_first():
            state = torch.load(self.resume_path, map_location="cpu")

        # 解包真实模块
        unet_raw    = self.accelerator.unwrap_model(self.pipe.unet)
        g_mlp_raw   = self.accelerator.unwrap_model(self.g_mlp)
        pos_abs_raw = self.accelerator.unwrap_model(self.pos_abs)
        src_type_raw= self.accelerator.unwrap_model(self.src_type)
        pre_ln_raw  = self.accelerator.unwrap_model(self.cond_pre_ln)
        adapter_raw = self.accelerator.unwrap_model(self.cond_adapter)

        # 1) 加载 UNet（LoRA 优先）
        if "unet_lora" in state and state["unet_lora"] is not None:
            set_peft_model_state_dict(unet_raw, state["unet_lora"])
            if self.accelerator.is_main_process:
                self.logger.info("Loaded LoRA weights into UNet.")
        elif "unet" in state:
            missing, unexpected = unet_raw.load_state_dict(state["unet"], strict=False)
            if self.accelerator.is_main_process:
                self.logger.warning(f"Loaded full UNet weights. Missing={len(missing)}, Unexpected={len(unexpected)}")

        # 2) 自定义模块
        g_mlp_raw.load_state_dict(state.get("g_mlp", {}), strict=False)
        pos_abs_raw.load_state_dict(state.get("pos_abs", {}), strict=False)
        src_type_raw.load_state_dict(state.get("src_type", {}), strict=False)
        pre_ln_raw.load_state_dict(state.get("cond_pre_ln", {}), strict=False)
        adapter_raw.load_state_dict(state.get("cond_adapter", {}), strict=False)

        # 3) 训练态 & 非 finetune 时才恢复优化器/调度器/epoch
        if self.mode == 'train' and not self.finetune:
            if "optimizer" in state:
                self.optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state:
                self.scheduler.load_state_dict(state["scheduler"])
            self.start_epoch = state.get("epoch", 0) + 1

        self.accelerator.wait_for_everyone()
        self.logger.info(f"Loaded checkpoint from {self.resume_path}", main_process_only=self.accelerator.is_main_process)




    def get_added_time_ids(self, batch_size=1):
        fps = 7 - 1  # As in SVD, fps-1
        dtype = torch.float16
        num_videos_per_prompt = 1

        added_time_ids = self.pipe._get_add_time_ids(
            fps,
            self.motion_bucket_id,
            self.noise_aug_strength,
            dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
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
    def encode_frames_by_clip(self, frames):
        """
        frames: shape [B, T, C, H, W], expected value range is between [0, 1].
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



        embeds = self.pipe._encode_image(flat, self.device, 1, self.do_classifier_free_guidance).reshape(B, T, -1)
        return embeds

    
    def get_frame_latents(self, frames_tensor: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Encode video frames to latents with optional masking.
        frames_tensor: [B, T or N, 3, H, W], expected value range is between [0, 1].
        mask: [B, T or N] bool, True means "encode this frame"; if None, treat as all True.
        Returns: [B, T or N, C_latent, h, w]  (unselected positions are zero-filled)
        """
        assert frames_tensor.dim() == 5, f"Expected [B,TN,3,H,W], got {tuple(frames_tensor.shape)}"
        min_value = float(frames_tensor.min())
        max_value = float(frames_tensor.max())
        assert min_value >= 0 and max_value <= 1, f"Expected input value range [0,1], got min {min_value}, max {max_value}"

        # print(f'frames_tensor.dtype: {frames_tensor.dtype}, frames_tensor.shape: {frames_tensor.shape}, min: {frames_tensor.min()}, max: {frames_tensor.max()}')

        B, T, C, H, W = frames_tensor.shape
        # 展平成(N=C*B*T, C, H, W)
        frames_btchw = frames_tensor.reshape(B*T, C, H, W)

        # 如果你要用 SVD 的 video_processor 做 resize/normalize：
        frames_btchw = self.pipe.video_processor.preprocess(
            frames_btchw, height=self.height, width=self.width
        )  # -> [B*T, 3, H', W']

        # 恢复回 [B, T, 3, H', W']
        frames_tensor = frames_btchw.view(B, T, 3, self.height, self.width)


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
        # 展平成 [B*T]，取被选中的全局下标
        flat_mask = mask.reshape(-1)
        flat_idx = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)  # [M]
        # 取被选中的帧并编码
        frames_flat = frames_tensor.reshape(B * TN, C, H, W)
        sel_frames = frames_flat.index_select(0, flat_idx)  # [M, 3, H, W]

        needs_upcast = (vae.dtype == torch.float16 and getattr(vae.config, "force_upcast", False))
        chunk_size = max(int(getattr(self, "chunk_size", 8)), 1)

        # 需要上采样到 fp32 时禁用 AMP；否则用 AMP（fp16/bf16）
        amp_ctx = nullcontext() if needs_upcast else self.accelerator.autocast(dtype=self.amp_dtype)

        if needs_upcast:
            vae.to(dtype=torch.float32)

        latents_sel = []
        with torch.no_grad(), amp_ctx:                
            for i in range(0, sel_frames.size(0), chunk_size):
                chunk = sel_frames[i:i + chunk_size].to(self.device, dtype=torch.float32, non_blocking=True)
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
            # print('latents_sel.dtype:', latents_sel.dtype) # torch.float16
            latents_out[flat_idx] = latents_sel.to(latents_out.dtype)
            
        if needs_upcast:
            vae.to(dtype=torch.float16)

        # 还原形状并做 scaling factor
        latents = latents_out.reshape(B, TN, latent_ch, h, w)
        # latents = latents * vae.config.scaling_factor
        return latents

    def decode_fn(self, latents, decode_chunk_size):
        ## to save memory, decode_chunk_size should be small
        video = self.pipe.decode_latents(latents, num_frames=latents.shape[1], decode_chunk_size=decode_chunk_size)  # Keep small chunk_size for finer memory control
        return video.permute(0, 2, 1, 3, 4).clamp(-1, 1)
    
    def build_image_embeddings(self, lr_frames, hd_frames=None, indices_mask=None,
                               spread_radius=0):
        """
        inputs:
            lr_frames: [B, T, 3, H, W], expected value range is between [0, 1].
            hd_frames: [B, N, 3, H, W], expected value range is between [0, 1].
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
        lr_token = self.encode_frames_by_clip(lr_frames)  # [B, T, D]
        base_dtype = lr_token.dtype
        D = lr_token.size(-1)

        tokens = lr_token.clone()
        attn_mask = torch.zeros(B, T, device=device, dtype=lr_token.dtype)  # [B,T], HD位置=1 其余=0

        # 2) 若没有 HD 或 mask 全 False，直接返回 LR
        has_hd = (
            hd_frames is not None and
            indices_mask is not None and
            indices_mask.numel() > 0 and
            bool(indices_mask.to(torch.bool).any())   
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
        hd_token = self.encode_frames_by_clip(hd_frames).to(base_dtype)       # [B,N,D]

        # 5) 掩码选择：显式成对索引，避免跨样本混合
        mask = indices_mask.bool().to(device)
        b_idx, s_idx = mask.nonzero(as_tuple=True)      # [M], [M]
        true_ind = indices[b_idx, s_idx]                       # [M], 一个batch中, 所有被选中的hd frame 的序号
        lr_sel = lr_token_at_ind[b_idx, s_idx, :]       # [M, D], 一个batch中, 所有被选中的hd frame 同样位置的 lr frame的特征
        hd_sel = hd_token[b_idx, s_idx, :]              # [M, D], 一个batch中, 所有被选中的hd frame 的特征

        # 6) 门控融合（标量门；需要向量门时可替换）
        feat = torch.cat([lr_sel, hd_sel, (hd_sel - lr_sel).abs()], dim=-1)  # [M, 3D]
        # print('Adaptive gate features:', feat.shape)
        g = torch.sigmoid(self.g_mlp(feat))                                  # [M,1] 或 [M,D]

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
        
        tokens_f32 = self.cond_pre_ln(tokens.float())           # LN 用 fp32 更稳
        tokens     = tokens + self.cond_adapter(tokens_f32)     # 残差注入


        tokens = tokens.to(self.unet_dtype)

        return tokens.contiguous(), attn_mask


    def compute_loss(self, lr_frames, hd_frames, indices_mask):
        '''
        input:
            lr_frames: tensor (B, T, 3, H, W), expected value range is between [0, 1].
            hd_frames: tensor (B, N, 3, H, W), expected value range is between [0, 1].
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
        t = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,  # e.g., 1000
                (B,), device=self.device, dtype=torch.long
            ) # [B]
        t_bt = t.view(B, 1, 1, 1, 1)  # 广播到时间维

        
        # 3) add noise
        noise = torch.randn_like(full_cond_latents, dtype=torch.float32).to(full_cond_latents.dtype)
        z_t   = self.noise_scheduler.add_noise(full_cond_latents, noise, t_bt)   # [B, T, C_latent, H_latent, W_latent]


        # 4) Get HD embeddings with pos (separate)
        image_embeddings, _ = self.build_image_embeddings(lr_frames, hd_frames, sparse_indices)
        # Get added time IDs
        added_time_ids = self.get_added_time_ids(batch_size=B)
        
        
        # 5) UNet 预测 velocity（保持 latent_dtype）
        latent_model_input = torch.cat([z_t, cond_lr_latents], dim=2)  # [B,T,2C,h,w]
        pred_v = self.pipe.unet(
            latent_model_input,
            t,  # [B]
            encoder_hidden_states=image_embeddings,
            added_time_ids=added_time_ids,
            return_dict=False
        )[0]  # [B,T,C,h,w]
        ### vel_pred: [B, T, C_latent, h, w], here is [B, 14, 4, 32, 32]
        # print('type of vel_pred:', vel_pred.dtype) #float16

        graph_zero = pred_v.sum() * 0.0

        

        # 6) compute x0 from predicted velocity
        alpha_bar = self.noise_scheduler.alphas_cumprod[t].to(device=self.device, dtype=torch.float32)  # [B]   # [B]      float32       
        ## 扩成 [B,1,1,1,1] 便于广播到 [B,T,C,h,w]
        sqrt_ab  = alpha_bar.view(B, 1, 1, 1, 1).sqrt()
        sqrt_oma = (1.0 - alpha_bar).view(B, 1, 1, 1, 1).sqrt()
        ## 用 v→x0 公式直接求 pred_original
        #    x0_hat = sqrt(ab)*x_t - sqrt(1-ab)*vel_pred
        pred_original = sqrt_ab * z_t.to(self.data_type) - sqrt_oma * pred_v.to(self.data_type)       # [B,T,C,h,w]
        pred_original = pred_original.to(latent_dtype)                           # ←★ 回到半精度


        # 7) 构造 v 的 GT（fp32）并计算 denoise loss（只在选中的槽位）
        # v-target & denoise loss —— 用 DDPM 的 get_velocity（v-pred）
        if self.noise_scheduler.config.prediction_type == "v_prediction":
            target_v = self.noise_scheduler.get_velocity(full_cond_latents, noise, t_bt)  # [B,T,C,h,w]
        else:
            target_v = noise
            
        ##  还原全时序 HD 掩码：hd_mask_full[b, t]=True 表示该样本在 t 是 HD 槽且 active
        hd_mask_full = torch.zeros(B, T, dtype=torch.bool, device=self.device)
        hd_mask_full.scatter_(1, sparse_indices, indices_mask)  # sparse_indices: [B,N] long, indices_mask: [B,N] bool; 
        ## hd_mask_full: [B,T]，True 表示该帧是激活的 HD 槽

        '''
        denoise loss 设计思路：
            HD 优先（样本内严格归一）：
                若样本有 k 个 HD 帧：
                每个 HD 帧权重 = α / k
                每个 LR 帧权重 = (1−α) / (T−k)
            若样本无 HD（k=0）：全帧均匀（1/T），不要额外弱化。
        '''

        k_hd = hd_mask_full.sum(dim=1)                    # [B]
        k_lr = (T - k_hd).clamp(min=1)                    # [B]
        has_hd = (k_hd > 0)

        alpha = 0.5

        # per-frame MSE，先在 (C,H,W) 上均值，再按帧加权
        per_frame_mse = ((pred_v.float() - target_v.float())**2).mean(dim=(2,3,4))  # [B,T]

        # 先构造 HD/LR 的系数，再按 mask 选择
        w_hd = torch.zeros(B, 1, device=self.device, dtype=torch.float32)
        w_lr = torch.zeros(B, 1, device=self.device, dtype=torch.float32)

        # 有 HD 的样本：HD/LR 按 α 分配；无 HD 的样本：均匀到每帧
        w_hd[has_hd] = (alpha / k_hd[has_hd]).unsqueeze(1)
        w_lr[has_hd] = ((1.0 - alpha) / k_lr[has_hd]).unsqueeze(1)

        # 无 HD：均匀分配到每帧（样本内和=1）
        w_uniform = torch.full((B, 1), 1.0 / T, device=self.device, dtype=torch.float32)

        w = torch.where(has_hd.unsqueeze(1), torch.where(hd_mask_full, w_hd, w_lr), w_uniform)  # [B,T]
        L_denoise = (per_frame_mse * w).sum(dim=1).mean()  # 标量（样本内和=1 → 尺度稳定）




        if self.use_latent_warp: ## to save memory, do not compute warp loss when not using it
            # 8) Decode to pixel space, no need to decode when using latent warp
            # with torch.no_grad():            
            #     gen_video = self.decode_fn(pred_original, decode_chunk_size = 14)  # 默认decode_chunk_size=14, [B, T, C_video, H_video, W_video]

            pred_original = pred_original.to(self.data_type)
            # 9) 计算光流损失
            L_temp = latent_warp_flow_loss(
            raft_model         = self.raft,
            pred_x0_latents_f32 = pred_original,   # [B,T,C_lat,hz,wz]
            frames_pixels           = lr_frames,     # [B,T,3,H,W], expected value range is between [0, 1].
            step                = 1,
            raft_scale          = 0.5
            )

            # 10) 计算重建loss
            if bool(indices_mask.any()):
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
            
            if bool(indices_mask.any()):
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
            L_temp = compute_temporal_loss(self.raft, gen_video, lr_frames, step = 1, scale = 0.5)


        # print('type of L_denoise:', L_denoise.dtype)
        # print('type of L_fid:', L_fid.dtype)
        # print('type of L_lr_temp:', L_lr_temp.dtype)
        # print('type of L_hd_temp:', L_hd_temp.dtype)

        # 11)  Total loss
        loss_sum = self.lamda_denoise * L_denoise + self.lamda_rec * L_rec + self.lamda_tempor * L_temp
        # loss = loss.to(self.data_type)  # Ensure same dtype as UNet
        print(f"Loss: L_denoise={L_denoise.item():.4f}, L_rec={L_rec.item():.4f}, L_temp={L_temp.item():.4f}")
        return loss_sum, L_denoise, L_rec, L_temp



    def training(self, epoch, dataloader, mode):
        
        self.unfrozenModules.train() # 这里面包含: unet, g_mlp, pos_abs, src_type
        total_loss, n_steps = 0.0, 0
        sum_l_denoise = 0.0
        sum_l_rec = 0.0
        sum_l_temp = 0.0
        
        epoch_loss_dict = {}

        
        total_steps = len(dataloader)  # Total steps in the epoch

        update_steps = 20  # Number of updates per epoch
        step_interval = max(total_steps // update_steps, 1)  # Calculate interval for updates


        progress_bar = tqdm(
            iterable=dataloader,
            total=total_steps,
            desc=f"{mode} Epoch {epoch + 1}/{self.num_train_epochs}",
            disable=not self.accelerator.is_local_main_process,
        )


        # 可选：开局清零
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(dataloader):
            ## LR, HR, sparse_hd_frames, sparse_hd_mask
            lr_frames, _, sparse_hd_frames, sparse_hd_mask = batch
            if step % step_interval == 0 or step == total_steps - 1:  # Update progress bar 50 times
                progress_bar.update(step_interval)
                
            with self.accelerator.accumulate(self.unfrozenModules):
                # AMP 由 Accelerate 统一管理；VAE 内部需要 fp32 的地方你在 compute_loss 里局部 autocast(False) 即可
                with self.accelerator.autocast():
                    loss_sum, L_denoise, L_rec, L_temp = self.compute_loss(lr_frames, sparse_hd_frames, sparse_hd_mask)  

                # 自动处理 GradScaler / DDP
                self.accelerator.backward(loss_sum)

                # 只在累积边界时做 step/clip/scheduler/zero_grad
                if self.accelerator.sync_gradients:
                    # （可选）梯度裁剪
                    # acc.clip_grad_norm_(model.parameters(), max_norm=getattr(self, "max_grad_norm", 1.0))

                    self.optimizer.step()
                    if getattr(self, "scheduler", None) is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

            # 统计全局平均 loss（所有进程）
            total_loss += self.accelerator.gather_for_metrics(loss_sum.detach()).mean().item()
            sum_l_denoise += self.accelerator.gather_for_metrics(L_denoise.detach()).mean().item()
            sum_l_rec += self.accelerator.gather_for_metrics(L_rec.detach()).mean().item()
            sum_l_temp += self.accelerator.gather_for_metrics(L_temp.detach()).mean().item()
            
            n_steps += 1

            if epoch == 0 and step == 0: # in GB
                allocated_memory_gb = torch.cuda.memory_allocated() / (1024**3)
                reserved_memory_gb = torch.cuda.memory_reserved() / (1024**3)
                self.logger.info(f'gpu memory allocated: {allocated_memory_gb:.2f} GB with batch size {self.train_batch_size}', main_process_only=self.accelerator.is_main_process)
                self.logger.info(f'gpu memory reserved: {reserved_memory_gb:.2f} GB with batch size {self.train_batch_size}', main_process_only=self.accelerator.is_main_process)


            if self.debug and step >= 5:
                break



        avg_total_loss = total_loss / max(1, n_steps)
        avg_l_denoise = sum_l_denoise / max(1, n_steps)
        avg_l_rec = sum_l_rec / max(1, n_steps)
        avg_l_temp = sum_l_temp / max(1, n_steps)
        # self.logger.info(f"Train epoch {epoch:3d} done. avg_total_loss={avg_total_loss:.6f}, avg_l_denoise={avg_l_denoise:.6f}, avg_l_rec={avg_l_rec:.6f}, avg_l_temp={avg_l_temp:.6f}, steps={n_steps}")
        # return avg_total_loss, avg_l_denoise, avg_l_rec, avg_l_temp
        
        epoch_loss_dict = {
            "sum_loss": avg_total_loss,
            "l_denoise": avg_l_denoise,
            "l_rec": avg_l_rec,
            "l_temp": avg_l_temp,
        }
        return epoch_loss_dict



    def test(self):
        self.unfrozenModules.eval() # 这里面包含: unet, g_mlp, pos_abs, src_type
        
        
    def run_training(self):
        self.load_checkpoint()
        
        train_loss_epoch_list = []

        is_best_epoch = False

        self.best_loss = float('inf')
        self.logger.info(f"Starting training from {self.start_epoch} epochs", main_process_only=self.accelerator.is_main_process)
        
        
        
        for epoch in range(self.start_epoch, self.num_train_epochs):
            if self.debug and epoch >= 3:
                break
            train_epoch_loss_dict = self.training(epoch, self.train_loader, mode='train')
            train_loss_epoch_list.append(train_epoch_loss_dict)
            sum_loss = train_epoch_loss_dict['sum_loss']
            l_denoise = train_epoch_loss_dict['l_denoise']
            l_rec = train_epoch_loss_dict['l_rec']
            l_temp = train_epoch_loss_dict['l_temp']
            


                
            if sum_loss < self.best_loss:
                self.best_loss = sum_loss
                is_best_epoch = True
            else:
                is_best_epoch = False

            self.save_checkpoint(epoch, is_best_epoch)
            
            # lr = self.optimizer.param_groups[0]['lr']
            lr = self.scheduler.get_last_lr()[0] if self.scheduler is not None else self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Epoch {epoch:3d} done. sum_loss={sum_loss:.6f}, l_denoise={l_denoise:.6f}, l_rec={l_rec:.6f}, l_temp={l_temp:.6f}, best_loss={self.best_loss:.6f}, lr={lr:.6f}", main_process_only=self.accelerator.is_main_process)
            
            
            visualize_loss(train_loss_epoch_list, 'train', os.path.join(self.logging_dir, 'loss_plot.jpg'), 
                                epoch=self.start_epoch, logger=self.logger)
                

        self.logger.info(f"\n{os.path.basename(__file__)}\n", main_process_only=self.accelerator.is_main_process)
        self.logger.info(f'log_path: {self.logging_dir}\n', main_process_only=self.accelerator.is_main_process)
        self.logger.info(f'output_ckpts_dir: {self.output_ckpts_dir}\n', main_process_only=self.accelerator.is_main_process)
        

if __name__ == "__main__":
    args = parse_args()
    worker = FewShotVideoSRWorker(args)

    runtime_mode = args.mode
    # main(args)
    if runtime_mode == 'train':
        worker.run_training()
    elif runtime_mode == 'test':
        worker.test()