import argparse
import os
import torch
from PIL import Image
from diffusers import DDIMScheduler
from controlnet.pipline_controlnet_xs_v2 import StableDiffusionPipelineXSv2
from controlnet.controlnetxs_appearance import StyleCodesModel
from diffusers.models import UNet2DConditionModel
from transformers import AutoProcessor, SiglipVisionModel



def use_stylecode(model,image_path, prompt,negative_prompt, num_inference_steps, stylecode,seed=None,image=None):
    # Load and preprocess image
    # Set up model components
    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=torch.float16, device="cuda")
    stylecodes_model = StyleCodesModel.from_unet(unet, size_ratio=1.0).to(dtype=torch.float16, device="cuda")


    print("running prompt = ",prompt, " negative_prompt = ",negative_prompt, " with code ", stylecode)
    stylecodes_model.load_model(model)

    pipe = StableDiffusionPipelineXSv2.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        unet=unet,
        stylecodes_model=stylecodes_model,
        torch_dtype=torch.float16,
        device="cuda",
        #scheduler=noise_scheduler,
        feature_extractor=None,
        safety_checker=None,
    )

    pipe.enable_model_cpu_offload()

    if image is None:
        image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))

    # Set up generator with a fixed seed for reproducibility
    if seed is not None and not -1:
        generator = torch.Generator(device="cuda").manual_seed(seed)
    else:
        generator = None

    # Run the image through the pipeline with the specified prompt
    output_images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=3,
        #image=image,
        num_inference_steps=num_inference_steps,
        generator=generator,
        controlnet_conditioning_scale=0.9,
        width=512,
        height=512,
        stylecode=stylecode,
    ).images
    return output_images


def process_single_image_both_ways(model,image_path, prompt, num_inference_steps,image=None):
    # Load and preprocess image
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

    stylecodes_model.load_model(model)

    pipe = StableDiffusionPipelineXSv2.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        unet=unet,
        stylecodes_model=stylecodes_model,
        torch_dtype=torch.float16,
        device="cuda",
        #scheduler=noise_scheduler,
        feature_extractor=None,
        safety_checker=None,
    )

    pipe.enable_model_cpu_offload()

    if image is None:
        image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))

    # Set up generator with a fixed seed for reproducibility
    seed = 238
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
        stylecode=None,
    ).images
    return output_images
    # Save the output image


def make_stylecode(model,image_path, image=None):
    
    # Set up model components
    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=torch.float16, device="cuda")
    stylecodes_model = StyleCodesModel.from_unet(unet, size_ratio=1.0).to(dtype=torch.float16, device="cuda")
    stylecodes_model.requires_grad_(False)
    stylecodes_model= stylecodes_model.to("cuda")   


    stylecodes_model.load_model(model)
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