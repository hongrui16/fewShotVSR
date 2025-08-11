import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image, export_to_video

import numpy as np
import os
import torch.nn as nn
import cv2
import argparse

from PIL import Image


pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16)
pipe.to("cuda")

prompt = "A man with short gray hair plays a red electric guitar."

web_img_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/guitar-man.png"

def main(args):
    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_path = args.image_path
    if not img_path is None and os.path.exists(img_path):

        basename = os.path.basename(img_path).split('.')[0]

        image = Image.open(img_path).convert("RGB")
    else:
        image = load_image(web_img_path)
        basename = os.path.basename(web_img_path).split('.')[0]
    height, width = image.size

    # if height > 512 or width > 512:
    #     new_height = min(height, 512)
    #     width = new_height * width // height
    #     height = new_height
    #     image = image.resize((width, height))

    # output = pipe(image=image, prompt=prompt).frames[0]
    output = pipe(image=image).frames[0]
    if prompt is not None:
        output = pipe(image=image, prompt=prompt).frames[0]
        video_path = os.path.join(output_dir, f"{basename}_prompt.mp4")
    else:
        output = pipe(image=image).frames[0]
        video_path = os.path.join(output_dir, f"{basename}.mp4")
    export_to_video(output, video_path)
    print(f"Video saved to {video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stable Video Diffusion")
    parser.add_argument("--image_path", type=str, default=None, help="Path to input image")
    parser.add_argument("--output", type=str, default= 'output', help="Path to save output video")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for video generation")
    args = parser.parse_args()

    # Load the input image

    main(args)

