import torch
from ..controlnet.attention_autoencoder import AttentionAutoencoder


class LatentAutoencoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_tensor": ("TENSOR", {}),
                "seq_w": ("INT", {"default": 16, "min": 1}),
                "seq_h": ("INT", {"default": 16, "min": 1}),
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR")
    FUNCTION = "encode_decode"
    CATEGORY = "Latent Nodes"

    def encode_decode(self, input_tensor, seq_w: int, seq_h: int):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        autoencoder = AttentionAutoencoder().to(device)

        # Encode
        latent = autoencoder.encode(input_tensor)

        # Decode
        reconstructed = autoencoder.decode(latent, seq_w, seq_h)

        return latent, reconstructed
