import argparse
import os
import torch
from PIL import Image
from diffusers import DDIMScheduler
from controlnet.pipline_controlnet_xs_v2 import StableDiffusionPipelineXSv2
from controlnet.controlnetxs_appearance import StyleCodesModel
from diffusers.models import UNet2DConditionModel
from transformers import AutoProcessor, SiglipVisionModel

def make_stylecode(image_path, image=None):
    
    # Set up model components
    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=torch.float16, device="cuda")
    stylecodes_model = StyleCodesModel.from_unet(unet, size_ratio=1.0).to(dtype=torch.float16, device="cuda")
    stylecodes_model.requires_grad_(False)
    stylecodes_model= stylecodes_model.to("cuda")   


    stylecodes_model.load_model("models/controlnet_model_11_80000.bin")
    # Load and preprocess image
    if image is None:
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="test_images/vangouh.jpg", help="Path to the input image")

    args = parser.parse_args()

    # Ensure the output folder exists
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)


    # Process a single image with the specified prompt
    print(make_stylecode(args.image_path))
