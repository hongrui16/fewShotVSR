import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image, export_to_video

import numpy as np
import os
import torch.nn as nn
import cv2
import argparse

from PIL import Image


pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt-1-1", torch_dtype=torch.float16)
pipe.to("cuda")

# prompt = "A man with short gray hair plays a red electric guitar."
# image = load_image(
#     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/guitar-man.png"
# )


def main(args):
    img_path = args.image_path
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image path {img_path} does not exist.")
    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    basename = os.path.basename(img_path).split('.')[0]

    image = Image.open(img_path).convert("RGB")

    height, width = image.size

    if height > 512 or width > 512:
        new_height = min(height, 512)
        width = new_height * width // height
        height = new_height
        image = image.resize((width, height), Image.LANCZOS)

    # output = pipe(image=image, prompt=prompt).frames[0]
    output = pipe(image=image).frames[0]
    export_to_video(output, os.path.join(output_dir, f"{basename}.mp4"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stable Video Diffusion")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default= 'output', help="Path to save output video")
    args = parser.parse_args()

    # Load the input image

    main(args)

