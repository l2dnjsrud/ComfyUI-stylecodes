import torch
from PIL import Image
from transformers import AutoProcessor, SiglipVisionModel
from ..controlnet.controlnetxs_appearance import StyleCodesModel
from diffusers.models import UNet2DConditionModel


class StyleCodeGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": "path/to/image.jpg", "multiline": False}),
                "model_path": ("STRING", {"default": "path/to/model.bin", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_stylecode"
    CATEGORY = "StyleCode Nodes"

    def generate_stylecode(self, image_path: str, model_path: str) -> str:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load models
        unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(device)
        stylecodes_model = StyleCodesModel.from_unet(unet, size_ratio=1.0).to(device)
        stylecodes_model.load_model(model_path)

        # Process image
        image = Image.open(image_path).convert("RGB").resize((512, 512))
        clip_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        clip_encoder = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224").to(device)
        clip_image = clip_processor(images=image, return_tensors="pt").pixel_values.to(device)
        clip_image_embeds = clip_encoder(pixel_values=clip_image, output_hidden_states=True).hidden_states[-2]

        # Generate style code
        stylecode = stylecodes_model.sref_autoencoder.make_stylecode(clip_image_embeds)
        return (stylecode,)
