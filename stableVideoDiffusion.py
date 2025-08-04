import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt-1-1", torch_dtype=torch.float16)
pipe.to("cuda")

prompt = "A man with short gray hair plays a red electric guitar."
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/guitar-man.png"
)

output = pipe(image=image, prompt=prompt).frames[0]
export_to_video(output, "output.mp4")