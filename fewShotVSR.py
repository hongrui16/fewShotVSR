from diffusers import StableVideoDiffusionPipeline
from diffusers import ControlNetModel
from peft import LoraConfig, get_peft_model  # PEFT library for LoRA
from transformers import CLIPVisionModel, CLIPProcessor

import torch

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
    torch_dtype=torch.float16
).to("cuda")

# Freeze VAE (encoder and decoder)
for param in pipe.vae.parameters():
    param.requires_grad = False
print("VAE frozen: Encoder/Decoder parameters not trainable.")




pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
    torch_dtype=torch.float16
).to("cuda")

# Apply LoRA to U-Net (rank=8 for small updates, alpha=4 for scaling)
lora_config = LoraConfig(
    r=8,  # Low rank (small parameters)
    lora_alpha=4,
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # Attention layers
    lora_dropout=0.1
)
pipe.unet = get_peft_model(pipe.unet, lora_config)
print("LoRA applied: Only ~1% of U-Net parameters trainable.")

# Fine-tune on 3 HR-LR pairs (custom dataset loader)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, pipe.unet.parameters()), lr=1e-5)
for epoch in range(100):  # 100-500 steps
    for hr_lr_pair in hr_lr_dataset:
        loss = compute_loss(pipe, hr_lr_pair)  # Your loss function
        loss.backward()
        optimizer.step()



class CustomControlNet(ControlNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_projector = torch.nn.Linear(512, 64)  # Project positional encodings

    def forward(self, latent, t, condition_map, pos_embed):
        # Standard ControlNet forward
        control_features = super().forward(latent, t, condition_map)
        # Add positional encoding
        pos_features = self.pos_projector(pos_embed)
        return control_features + pos_features.repeat(control_features.shape[0], 1, 1, 1)

# Load and replace
custom_controlnet = CustomControlNet.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16).to("cuda")
pipe.controlnet = custom_controlnet


clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def get_condition_map(hd_frame):
    hd_image = Image.fromarray(hd_frame).resize((224, 224))
    inputs = clip_processor(images=hd_image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        embedding = clip_model(**inputs).last_hidden_state  # [1, seq_len, 768]
    return embedding  # Inject via custom ControlNet

# Positional encoding for indices (like t)
def positional_encoding(indices, max_pos=100, d_model=512):
    pe = torch.zeros(len(indices), d_model).to("cuda")
    position = torch.tensor(indices, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model)).to("cuda")
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # [1, num_hd, d_model]


hd_indices = [0, 5, 10]
pos_embed = positional_encoding(hd_indices)
condition_map = get_condition_map(hd_frames[0])  # For nearest HD
generated = pipe.unet(lr_latent, t, condition_map, pos_embed)