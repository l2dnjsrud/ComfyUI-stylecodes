import torch
from diffusers import DDIMScheduler
from ..controlnet.pipline_controlnet_xs_v2 import StableDiffusionPipelineXSv2
from ..controlnet.controlnetxs_appearance import StyleCodesModel
from diffusers.models import UNet2DConditionModel


class StyleCodeProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "Describe the image", "multiline": False}),
                "stylecode": ("STRING", {"default": "style_code_here", "multiline": False}),
                "model_path": ("STRING", {"default": "path/to/model.bin", "multiline": False}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "StyleCode Nodes"

    def generate_image(self, prompt: str, stylecode: str, model_path: str, steps: int):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load models
        unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(device)
        stylecodes_model = StyleCodesModel.from_unet(unet, size_ratio=1.0).to(device)
        stylecodes_model.load_model(model_path)

        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
        )

        pipe = StableDiffusionPipelineXSv2(
            unet=unet,
            stylecodes_model=stylecodes_model,
            scheduler=scheduler,
            device=device,
        )

        generator = torch.manual_seed(42)

        output_images = pipe(
            prompt=prompt,
            stylecode=stylecode,
            guidance_scale=7.5,
            num_inference_steps=steps,
            generator=generator,
        ).images
        return (output_images[0],)
