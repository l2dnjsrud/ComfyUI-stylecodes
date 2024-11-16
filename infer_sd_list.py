import argparse
import cv2
import numpy as np
import torch
from controlnet.controlnetxs_appearance import StyleCodesModel
from PIL import Image, ImageDraw, ImageFont
from controlnet.pipline_controlnet_xs_v2 import StableDiffusionPipelineXSv2
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.utils import load_image
from diffusers import DDIMScheduler
import datetime
import os
from copy import deepcopy
from safetensors import safe_open

from controlnet_aux.processor import Processor
import random
def image_grid(imgs, rows, cols, prompt):
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*(h+30)))  # Extra 30 pixels for text
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*(h+30)))
    
    # Add labels
    draw.text((10, 5), f"Prompt: {prompt}", fill="white", font=font)

    return grid

parser = argparse.ArgumentParser()
parser.add_argument("--negative_prompt", type=str, default="low quality, bad quality, sketches")
parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
parser.add_argument("--num_inference_steps", type=int, default=20)
parser.add_argument("--input_folder", type=str, default="test_images", help="Path to the folder containing input images")
parser.add_argument("--output_folder", type=str, default="output", help="Path to the folder to save output images")
#SG161222/Realistic_Vision_V4.0_noVAE
#Linaqruf/anything-v3.0
args = parser.parse_args()
processor_id = 'lineart_realistic'
canny_processor = Processor(processor_id)

image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_ckpt = "models/controlnet_model_11_80000.bin"
device = "cuda"

unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=torch.float16, device="cuda")
stylecodes_model = StyleCodesModel.from_unet(unet, size_ratio=1.0).to(dtype=torch.float16, device="cuda")
#SG161222/Realistic_Vision_V5.1_noVAE
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

stylecodes_model.load_model(ip_ckpt)



pipe = StableDiffusionPipelineXSv2.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    unet=unet,
    stylecodes_model=stylecodes_model,
    torch_dtype=torch.float16,
    device="cuda",
    scheduler=noise_scheduler,
    feature_extractor=None,
    safety_checker=None,
)

pipe.enable_model_cpu_offload()

distance_conds= [-1]
prompts= ["a man close up","a woman doctor in a room portrait", "a cow close up","a bottle"]
num_samples=4
def process_image(image_path,   num_inference_steps):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))
    seed = 235
    all_images = [image]  # Start with the original image
    for prompt in prompts:
        seed = seed + 1
        generator = torch.Generator(unet.device).manual_seed(seed)      
        images = pipe(
            prompt=prompt,
            guidance_scale=3,
            image=image,
            num_inference_steps=num_inference_steps,
            generator=generator,
            controlnet_conditioning_scale=0.9,
            width=512,
            height=512,
        ).images
        all_images.extend(images)
    
    return all_images

# Ensure the output folder exists
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

# Process all images in the input folder
all_processed_images = []
image_files = [f for f in os.listdir(args.input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

for filename in image_files:
    input_path = os.path.join(args.input_folder, filename)
    print(f"Processing {filename}... as  {os.path.splitext(filename)[0]}")    
    images = process_image(input_path, args.num_inference_steps)
    all_processed_images.extend(images)

# Create and save the final grid with all images
rows = len(image_files)
cols = len(prompts) + 1  # +1 for the original image
final_grid = image_grid(all_processed_images, rows, cols, "Varies per image")
final_save_path = os.path.join(args.output_folder, f"final_grid_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.png")

print("Saving final grid...")
final_grid.save(final_save_path)
print(f"Final grid saved to {final_save_path}")

print("All images processed.")