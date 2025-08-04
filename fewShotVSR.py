import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import StableVideoDiffusionPipeline, ControlNetModel, DPMSolverMultistepScheduler
from peft import LoraConfig, get_peft_model
from transformers import CLIPVisionModel, CLIPProcessor
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from torchvision.models.optical_flow import raft_small
import numpy as np
from PIL import Image

class CustomControlNet(ControlNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_projector = nn.Linear(512, 64)  # Project positional encodings

    def forward(self, latent, t, condition_map, pos_embed):
        control_features = super().forward(latent, t, condition_map)
        pos_features = self.pos_projector(pos_embed)
        pos_features = pos_features.view(pos_features.size(0), pos_features.size(1), 1, 1).expand_as(control_features)
        return control_features + pos_features


class FewShotVideoSRTrainer:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"

        # Load SVD pipeline
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
            torch_dtype=torch.float16
        ).to(self.device)

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

        # Custom ControlNet
        self.pipe.controlnet = CustomControlNet.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16
        ).to(self.device)

        # CLIP for condition map
        self.clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        # LPIPS
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(self.device)

        # RAFT for temporal loss
        self.raft = raft_small(pretrained=True).eval().to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.pipe.unet.parameters()), lr=1e-5)

        # Scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

    def positional_encoding(self, indices, max_pos=100, d_model=512):
        pe = torch.zeros(len(indices), d_model).to(self.device)
        position = torch.tensor(indices, dtype=torch.float32).unsqueeze(1).to(self.device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model)).to(self.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def get_condition_map(self, hd_frame):
        hd_image = Image.fromarray(hd_frame).resize((224, 224))
        inputs = self.clip_processor(images=hd_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embedding = self.clip_model(**inputs).last_hidden_state
        return embedding

    def compute_loss(self, lr_frame, hd_frame, condition_map, pos_embed, t):
        lr_tensor = torch.from_numpy(lr_frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        hd_tensor = torch.from_numpy(hd_frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0

        lr_latent = self.pipe.vae.encode(lr_tensor).latent_dist.sample()
        hd_latent = self.pipe.vae.encode(hd_tensor).latent_dist.sample()

        noise = torch.randn_like(lr_latent)
        z_t = self.pipe.scheduler.add_noise(lr_latent, noise, t)

        predicted_noise = self.pipe.unet(z_t, t, encoder_hidden_states=condition_map, added_cond_kwargs={"pos_embed": pos_embed})

        L_denoise = nn.MSELoss()(predicted_noise, noise)
        generated_latent = self.pipe.scheduler.step(predicted_noise, t, z_t).pred_original_sample
        generated_frame = self.pipe.vae.decode(generated_latent).sample[0]

        L_fid = nn.L1Loss()(generated_frame, hd_tensor[0])
        L_perc = self.lpips(generated_frame.unsqueeze(0), hd_tensor[0].unsqueeze(0))
        L_temp = 0.0  # Temporal loss placeholder

        return L_denoise + 0.5 * L_fid + 0.5 * L_perc + 0.1 * L_temp

    def train(self, dataset, epochs=100):
        t = torch.tensor([10]).to(self.device)
        for epoch in range(epochs):
            for lr, hd in dataset:
                condition_map = self.get_condition_map(hd)
                pos_embed = self.positional_encoding([0, 5, 10])
                loss = self.compute_loss(lr, hd, condition_map, pos_embed, t)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

        print("Training complete.")


# Dummy dataset
hr_lr_dataset = [
    (np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
     np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    for _ in range(3)
]

trainer = FewShotVideoSRTrainer(device="cuda")
trainer.train(hr_lr_dataset, epochs=10)
