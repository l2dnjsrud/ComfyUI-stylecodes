import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import datetime
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.normalization import GroupNorm
import base64
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class AttentionAutoencoder(nn.Module):
    def __init__(self, input_dim=768,output_dim=1280, d_model=512, latent_dim=20, seq_len=196, num_heads=4, num_layers=3, out_intermediate=512):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_dim = input_dim  # Adjusted to 768
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.seq_len = seq_len  # Adjusted to 196
        self.out_intermediate = out_intermediate
        self.output_dim = output_dim

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Input Projection (adjusted to project from input_dim=768 to d_model=512)
        self.input_proj = nn.Linear(input_dim, d_model)

        # Latent Initialization
        self.latent_init = nn.Parameter(torch.randn(1, d_model))

        # Cross-Attention Encoder
        self.num_layers = num_layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        # Latent Space Refinement
        self.latent_proj = nn.Linear(d_model, latent_dim)
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.latent_to_d_model = nn.Linear(latent_dim, d_model)

        # Mapping latent to intermediate feature map
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, batch_first=True),
            num_layers=2
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, output_dim)
        self.tgt_init = nn.Parameter(torch.randn(1, d_model))



    def encode(self, src):
        # src shape: [batch_size, seq_len (196), input_dim (768)]
        batch_size, seq_len, input_dim = src.shape
    
        # Project input_dim (768) to d_model (512)
        src = self.input_proj(src)  # Shape: [batch_size, seq_len (196), d_model (512)]
        src = self.pos_encoder(src)  # Add positional encoding
        
        # Latent initialization
        latent = self.latent_init.repeat(batch_size, 1).unsqueeze(1)  # Shape: [batch_size, 1, d_model]
        
        # Cross-attend latent with input sequence
        for i in range(self.num_layers):
            latent, _ = self.attention_layers[i](latent, src, src)
        
        # Project to latent dimension and normalize
        latent = self.latent_proj(latent.squeeze(1))  # Shape: [batch_size, latent_dim]
        latent = self.latent_norm(latent)
    
        return latent
    
    def decode(self, latent, seq_w, seq_h):
        batch_size = latent.size(0)
        
        target_seq_len = seq_w * seq_h

        # Project latent_dim back to d_model
        memory = self.latent_to_d_model(latent).unsqueeze(1)  # Shape: [batch_size, 1, d_model]

        # Target initialization
        # Repeat the learned target initialization to match the target sequence length
        tgt = self.tgt_init.repeat(batch_size, target_seq_len, 1)  # Shape: [batch_size, target_seq_len, d_model]

        # Apply positional encoding
        tgt = self.pos_encoder(tgt)

        # Apply transformer decoder
        output = self.transformer_decoder(tgt, memory)  # Shape: [batch_size, target_seq_len, d_model]

        # Project to output_dim
        output = self.output_proj(output)  # Shape: [batch_size, target_seq_len, output_dim]

        # Reshape output to (batch_size, seq_w, seq_h, output_dim)
        output = output.view(batch_size, seq_w, seq_h, self.output_dim)

        # Permute dimensions to (batch_size, output_dim, seq_w, seq_h)
        output = output.permute(0, 3, 1, 2)  # Shape: [batch_size, output_dim, seq_w, seq_h]

        return output
    
    def forward(self, src, seq_w, seq_h):
        latent = self.encode(src)
        output = self.decode(latent, seq_w, seq_h)
        return output

    def encode_to_base64(self, latent_vector, bits_per_element):
        max_int = 2 ** bits_per_element - 1
        q_latent = ((latent_vector + 1) * (max_int / 2)).clip(0, max_int).astype(np.uint8)
        byte_array = q_latent.tobytes()
        encoded_string = base64.b64encode(byte_array).decode('utf-8')
        # Remove padding characters
        return encoded_string.rstrip('=')

    def decode_from_base64(self, encoded_string, bits_per_element, latentdim):
   
        # Add back padding if it's missing
        missing_padding = len(encoded_string) % 4
        if missing_padding:
            encoded_string += '=' * (4 - missing_padding)
        byte_array = base64.b64decode(encoded_string)
        q_latent = np.frombuffer(byte_array, dtype=np.uint8)[:latentdim]
        max_int = 2 ** bits_per_element - 1
        latent_vector = q_latent.astype(np.float32) * 2 / max_int - 1
        return latent_vector

    def forward_encoding(self, src, seq_w, seq_h):
        """
        Encodes the input `src` into a latent representation, encodes it to a Base64 string,
        decodes it back to the latent space, and then decodes it to the output.
        
        Args:
            src: The input data to encode.
        
        Returns:
            output: The decoded output from the latent representation.
        """
        # Step 1: Encode the input to latent space
        latent = self.encode(src)  # latent is of shape (batch_size, self.latentdim)
        batch_size, latentdim = latent.shape
        
        # Ensure bits_per_element is appropriate
        bits_per_element = int(120 / latentdim)  # Example: latentdim = 20, bits_per_element = 6
        if bits_per_element > 8:
            raise ValueError("bits_per_element cannot exceed 8 when using uint8 for encoding.")
        
        encoded_strings = []
        
        # Step 2: Encode each latent vector to a Base64 string
        for i in range(batch_size):
            latent_vector = latent[i].cpu().numpy()
            encoded_string = self.encode_to_base64(latent_vector, bits_per_element)
            encoded_strings.append(encoded_string)
        
        decoded_latents = []
        
        # Step 3: Decode each Base64 string back to the latent vector
        for i, encoded_string in enumerate(encoded_strings):
            print(encoded_string)
            decoded_latent = self.decode_from_base64(encoded_string, bits_per_element, latentdim)
            decoded_latents.append(decoded_latent)
        
        # Step 4: Convert the list of decoded latents back to a tensor
        decoded_latents = torch.tensor(decoded_latents, dtype=latent.dtype, device=latent.device)
        
        # Step 5: Decode the latent tensor into the output
        output = self.decode(decoded_latents,seq_w, seq_h)
        
        return output, encoded_strings
    
    def forward_from_stylecode (self, stylecode, seq_w, seq_h,dtyle,device):

        latentdim = 20
        bits_per_element = 6
        decoded_latents = []

        
        #for i, encoded_string in enumerate(stylecode):
        decoded_latent = self.decode_from_base64(stylecode, bits_per_element, latentdim)
        decoded_latents.append(decoded_latent)
        
        # Step 4: Convert the list of decoded latents back to a tensor
        decoded_latents = torch.tensor(decoded_latents, dtype=dtyle, device=device)

        output = self.decode(decoded_latents, seq_w, seq_h)
        return output
    
    @torch.no_grad()
    def make_stylecode (self,src):
        src = src.to("cuda")
        self = self.to("cuda")
        print(src.device,self.device,self.input_proj.weight.device)
        latent = self.encode(src)  # latent is of shape (batch_size, self.latentdim)
        batch_size, latentdim = latent.shape
        
        # Ensure bits_per_element is appropriate
        bits_per_element = int(120 / latentdim)  # Example: latentdim = 20, bits_per_element = 6
        if bits_per_element > 8:
            raise ValueError("bits_per_element cannot exceed 8 when using uint8 for encoding.")
        
        encoded_strings = []
        
        # Step 2: Encode each latent vector to a Base64 string
        for i in range(batch_size):
            latent_vector = latent[i].cpu().numpy()
            encoded_string = self.encode_to_base64(latent_vector, bits_per_element)
            encoded_strings.append(encoded_string)
        return encoded_strings