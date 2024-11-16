import argparse
import os
import torch
from PIL import Image
from diffusers import DDIMScheduler
from controlnet.pipline_controlnet_xs_v2 import StableDiffusionPipelineXSv2
from controlnet.controlnetxs_appearance import StyleCodesModel
from diffusers.models import UNet2DConditionModel
from transformers import AutoProcessor, SiglipVisionModel

def process_single_image(image_path, prompt, num_inference_steps, output_folder,stylecode):
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))

    # Set up generator with a fixed seed for reproducibility
    seed = 238
    clip_image_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    image_encoder = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224").to(dtype=torch.float16,device=stylecodes_model.device)
    clip_image = clip_image_processor(images=image, return_tensors="pt").pixel_values
    clip_image = clip_image.to(stylecodes_model.device, dtype=torch.float16)
    clip_image = {"pixel_values": clip_image}
    clip_image_embeds = image_encoder(**clip_image, output_hidden_states=True).hidden_states[-2]

    # Run the image through the pipeline with the specified prompt
    code = stylecodes_model.sref_autoencoder.make_stylecode(clip_image_embeds)
    print("stylecode = ",code)
    return code




# Set up argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, default="test_images/vangouh.jpg", help="Path to the input image")
parser.add_argument("--prompt", type=str, default="a man in a room", help="Prompt for image generation")
parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps")
parser.add_argument("--output_folder", type=str, default="output", help="Path to save the output image")
#stylecode
parser.add_argument("--stylecode", type=str, default=r"Aj8dPwAACD8rECAlCC8UGzU/HwI", help="")


args = parser.parse_args()

# Ensure the output folder exists
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

# Set up model components
unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=torch.float16, device="cuda")
stylecodes_model = StyleCodesModel.from_unet(unet, size_ratio=1.0).to(dtype=torch.float16, device="cuda")
stylecodes_model.requires_grad_(False)
stylecodes_model= stylecodes_model.to("cuda")   


stylecodes_model.load_model("models/controlnet_model_11_80000.bin")

# Process a single image with the specified prompt
process_single_image(args.image_path, args.prompt, args.num_inference_steps, args.output_folder,stylecode=args.stylecode)
