import argparse
import os
import torch
from PIL import Image
from diffusers import DDIMScheduler
from controlnet.pipline_controlnet_xs_v2 import StableDiffusionPipelineXSv2
from controlnet.controlnetxs_appearance import StyleCodesModel
from diffusers.models import UNet2DConditionModel

def process_single_image(image_path, prompt, num_inference_steps, output_folder):
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))

    # Set up generator with a fixed seed for reproducibility
    seed = 236
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Run the image through the pipeline with the specified prompt
    output_images = pipe(
        prompt=prompt,
        guidance_scale=3,
        image=image,
        num_inference_steps=num_inference_steps,
        generator=generator,
        controlnet_conditioning_scale=0.9,
        width=512,
        height=512,
    ).images

    # Save the output image
    output_path = os.path.join(output_folder, f"processed_{os.path.basename(image_path)}")
    output_images[0].save(output_path)
    print(f"Processed image saved to {output_path}")

# Set up argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, default="test_images/bike.png", help="Path to the input image")
parser.add_argument("--prompt", type=str, default="a man in a room", help="Prompt for image generation")
parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps")
parser.add_argument("--output_folder", type=str, default="output", help="Path to save the output image")

args = parser.parse_args()

# Ensure the output folder exists
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

# Set up model components
unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=torch.float16, device="cuda")
stylecodes_model = StyleCodesModel.from_unet(unet, size_ratio=1.0).to(dtype=torch.float16, device="cuda")

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

stylecodes_model.load_model("models/controlnet_model_11_80000.bin")

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

# Process a single image with the specified prompt
process_single_image(args.image_path, args.prompt, args.num_inference_steps, args.output_folder)
