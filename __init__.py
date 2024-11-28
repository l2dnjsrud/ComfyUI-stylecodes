from .nodes.stylecode_generator_node import StyleCodeGenerator
from .nodes.stylecode_processor_node import StyleCodeProcessor
from .nodes.latent_autoencoder_node import LatentAutoencoder

NODE_CLASS_MAPPINGS = {
    "Style Code Generator": StyleCodeGenerator,
    "Style Code Processor": StyleCodeProcessor,
    "Latent Autoencoder": LatentAutoencoder,
}

print("Custom ComfyUI Nodes: Loaded")
