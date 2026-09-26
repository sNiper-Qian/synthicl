# Description: This file contains utility functions that are used across the project.
from torch.nn import functional as F
from torch import Tensor
import torch
from typing import Callable
import numpy as np  
import math
from torch import nn
from icil.policy.latte import UnconditionalTransformerBlock, TransformerBlock, get_2d_sincos_pos_embed, get_1d_sincos_temp_embed, modulate
from timm.models.vision_transformer import Mlp, PatchEmbed
from einops import rearrange, repeat

def get_activation_fn(activation: str) -> Callable:
    """Returns the activation function given its name."""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return F.tanh
    elif activation == "sigmoid":
        return F.sigmoid
    else:
        raise ValueError(f"Activation function {activation} not supported.")

def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()

def move_to_device(data: dict, device: torch.device, dtype: torch.dtype|None = torch.float32) -> dict:
    for k, v in data.items():
        if torch.is_tensor(v):
            data[k] = v.to(device)
            if dtype is not None:
                data[k] = data[k].to(dtype)
        else:
            data[k] = v
    return data

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)

        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        # Add position encoding
        x = x + self.pe[:, :seq_len, :]
        return x

class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        batch_first: bool = True,  # new argument
    ):
        """
        A single Transformer Decoder layer without self-attention.
        Consists of:
         - Cross-attention sub-layer
         - Feed-forward sub-layer
         - Residual connections and LayerNorm in each sub-layer

        Args:
            d_model:         Dimension of embeddings.
            nhead:           Number of attention heads.
            dim_feedforward: Hidden layer size in the feed-forward network.
            dropout:         Dropout probability.
            batch_first:     If True, expects input shape (B, T, E).
                             If False, expects input shape (T, B, E).
        """
        super().__init__()
        
        # Cross-Attention: query=decoder input, key/value=memory (encoder output)
        # We pass batch_first=batch_first to align shapes accordingly:
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first
        )

        # LayerNorms for the two sub-layers (cross-attn + feed-forward)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network (simple 2-layer MLP)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

        self.batch_first = batch_first

    def forward(
        self,
        tgt: torch.Tensor,            # (batch_size, tgt_len, d_model) if batch_first=True
        memory: torch.Tensor,         # (batch_size, src_len, d_model) if batch_first=True
        memory_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
    ):
        """
        Args:
            tgt:
                Decoder input embeddings. Shape depends on batch_first:
                  - (B, T, d_model) if batch_first=True
                  - (T, B, d_model) if batch_first=False
            memory:
                Encoder output / "memory". Shape depends on batch_first:
                  - (B, S, d_model) if batch_first=True
                  - (S, B, d_model) if batch_first=False
            memory_mask:
                Optional mask of shape (T, S) if batch_first=True, or (T, S) swapped if not.
            memory_key_padding_mask:
                Optional mask of shape (B, S) indicating padding in memory.

        Returns:
            out:
                Updated decoder embeddings, same shape as `tgt`.
        """
        # ---- Cross-Attention Sub-layer ----
        # Q = tgt, K = memory, V = memory
        x_attn_out, _ = self.cross_attn(
            query=tgt,
            key=memory,
            value=memory,
            attn_mask=memory_mask,                 # shape depends on batch_first
            key_padding_mask=memory_key_padding_mask
        )
        # Residual + LayerNorm
        tgt2 = self.norm1(tgt + self.dropout(x_attn_out))

        # ---- Feed-Forward Sub-layer ----
        ffn_out = self.ffn(tgt2)
        out = self.norm2(tgt2 + self.dropout(ffn_out))

        return out

class SpatialTemperalSelfAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        input_size=96,
        patch_size=16,
        in_channels=3,
        num_frames=8,
        dropout: float = 0.1,
        batch_first: bool = True,  # new argument
    ):
        """
        A single Transformer Decoder layer with a spatial-attention and a cross-attention layer.
        Consists of:
         - Spatial attention sub-layer
         - Cross-attention sub-layer
         - Feed-forward sub-layer
         - Residual connections and LayerNorm in each sub-layer

        Args:
            d_model:         Dimension of embeddings.
            nhead:           Number of attention heads.
            dim_feedforward: Hidden layer size in the feed-forward network.
            dropout:         Dropout probability.
            batch_first:     If True, expects input shape (B, T, E).
                             If False, expects input shape (T, B, E).
        """
        super().__init__()
        # Spatial-Attention: query=decoder input, key/value=decoder input
        self.spatial_attn = UnconditionalTransformerBlock(
            hidden_size=d_model,
            num_heads=nhead,
            mlp_ratio=int(dim_feedforward / d_model),
        )
        self.temp_attn = UnconditionalTransformerBlock(
            hidden_size=d_model,
            num_heads=nhead,
            mlp_ratio=int(dim_feedforward / d_model),
        )
        # Cross-Attention: query=decoder input, key/value=memory (encoder output)
        # We pass batch_first=batch_first to align shapes accordingly:

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first
        )

        # LayerNorms for the two sub-layers (cross-attn + feed-forward)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network (simple 2-layer MLP)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

        self.batch_first = batch_first
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, d_model)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, d_model), requires_grad=False)
        temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], self.temp_embed.shape[-2])
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

    def forward(
        self,
        tgt: torch.Tensor,            # (batch_size, tgt_len, C, H, W) if batch_first=True
        memory: torch.Tensor,         # (batch_size, src_len, C, H, W) if batch_first=True
        memory_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
        first_layer: bool = False
    ):
        """
        Args:
            tgt:
                Decoder input embeddings. Shape depends on batch_first:
                  - (B, T, d_model) if batch_first=True
                  - (T, B, d_model) if batch_first=False
            memory:
                Encoder output / "memory". Shape depends on batch_first:
                  - (B, S, d_model) if batch_first=True
                  - (S, B, d_model) if batch_first=False
            memory_mask:
                Optional mask of shape (T, S) if batch_first=True, or (T, S) swapped if not.
            memory_key_padding_mask:
                Optional mask of shape (B, S) indicating padding in memory.

        Returns:
            out:
                Updated decoder embeddings, same shape as `tgt`.
        """
        batches = tgt.shape[0]
        if first_layer:
            tgt = rearrange(tgt, 'b f c h w -> (b f) c h w')
            tgt = self.x_embedder(tgt) + self.pos_embed
        # ---- Spatial-Attention Sub-layer ----
        if len(memory.shape) == 4:
            memory = rearrange(memory, 'b f t d -> b (f t) d', b=batches)
        tgt = self.spatial_attn(tgt)
        # ----- Temporal-Attention Sub-layer ----
        tgt = rearrange(tgt, '(b f) t d -> (b t) f d', b=batches)
        if first_layer:
            tgt = tgt + self.temp_embed 
        tgt = self.temp_attn(tgt)
        tgt = rearrange(tgt, '(b t) f d -> b (f t) d', b=batches)
        # ---- Feed-Forward Sub-layer ----
        ffn_out = self.ffn(tgt)
        out = self.norm2(tgt + self.dropout(ffn_out))
        return out

class SpatialTemperalCrossAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        input_size=96,
        patch_size=16,
        in_channels=3,
        num_frames=8,
        dropout: float = 0.1,
        batch_first: bool = True,  # new argument
    ):
        """
        A single Transformer Decoder layer with a spatial-attention and a cross-attention layer.
        Consists of:
         - Spatial attention sub-layer
         - Cross-attention sub-layer
         - Feed-forward sub-layer
         - Residual connections and LayerNorm in each sub-layer

        Args:
            d_model:         Dimension of embeddings.
            nhead:           Number of attention heads.
            dim_feedforward: Hidden layer size in the feed-forward network.
            dropout:         Dropout probability.
            batch_first:     If True, expects input shape (B, T, E).
                             If False, expects input shape (T, B, E).
        """
        super().__init__()
        # Spatial-Attention: query=decoder input, key/value=decoder input
        self.spatial_attn = UnconditionalTransformerBlock(
            hidden_size=d_model,
            num_heads=nhead,
            mlp_ratio=int(dim_feedforward / d_model),
        )
        self.temp_attn = UnconditionalTransformerBlock(
            hidden_size=d_model,
            num_heads=nhead,
            mlp_ratio=int(dim_feedforward / d_model),
        )
        # Cross-Attention: query=decoder input, key/value=memory (encoder output)
        # We pass batch_first=batch_first to align shapes accordingly:

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first
        )

        # LayerNorms for the two sub-layers (cross-attn + feed-forward)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network (simple 2-layer MLP)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

        self.batch_first = batch_first
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, d_model)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, d_model), requires_grad=False)
        temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], self.temp_embed.shape[-2])
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

    def forward(
        self,
        tgt: torch.Tensor,            # (batch_size, tgt_len, C, H, W) if batch_first=True
        memory: torch.Tensor,         # (batch_size, src_len, C, H, W) if batch_first=True
        memory_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
        first_layer: bool = False
    ):
        """
        Args:
            tgt:
                Decoder input embeddings. Shape depends on batch_first:
                  - (B, T, d_model) if batch_first=True
                  - (T, B, d_model) if batch_first=False
            memory:
                Encoder output / "memory". Shape depends on batch_first:
                  - (B, S, d_model) if batch_first=True
                  - (S, B, d_model) if batch_first=False
            memory_mask:
                Optional mask of shape (T, S) if batch_first=True, or (T, S) swapped if not.
            memory_key_padding_mask:
                Optional mask of shape (B, S) indicating padding in memory.

        Returns:
            out:
                Updated decoder embeddings, same shape as `tgt`.
        """
        batches = tgt.shape[0]
        if first_layer:
            tgt = rearrange(tgt, 'b f c h w -> (b f) c h w')
            tgt = self.x_embedder(tgt) + self.pos_embed
        # ---- Spatial-Attention Sub-layer ----
        if len(memory.shape) == 4:
            memory = rearrange(memory, 'b f t d -> b (f t) d', b=batches)
        tgt = self.spatial_attn(tgt)
        # ----- Temporal-Attention Sub-layer ----
        tgt = rearrange(tgt, '(b f) t d -> (b t) f d', b=batches)
        if first_layer:
            tgt = tgt + self.temp_embed 
        tgt = self.temp_attn(tgt)
        # ---- Cross-Attention Sub-layer ----
        tgt = rearrange(tgt, '(b t) f d -> b (f t) d', b=batches)
        # Q = tgt, K = memory, V = memory
        x_attn_out, _ = self.cross_attn(
            query=tgt,
            key=memory,
            value=memory,
            attn_mask=memory_mask,                 # shape depends on batch_first
            key_padding_mask=memory_key_padding_mask
        )
        # Residual + LayerNorm
        tgt2 = self.norm1(tgt + self.dropout(x_attn_out))

        # ---- Feed-Forward Sub-layer ----
        ffn_out = self.ffn(tgt2)
        out = self.norm2(tgt2 + self.dropout(ffn_out))
        return out


class SpatialTemperalSelfAttentionEncoder(nn.Module):
    def __init__(
        self,
        d_model=32,
        nhead=4,
        num_layers=4,
        dim_feedforward=128,
        dropout=0.1,
        patch_size=16,
        in_channels=3,
        input_size=96,
        num_frames=8,
        batch_first=True,
    ):
        """
        A stack of 'num_layers' SpatialCrossAttentionLayer layers.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            SpatialTemperalSelfAttentionLayer(d_model=d_model, 
                                       nhead=nhead, 
                                       dim_feedforward=dim_feedforward, 
                                       dropout=dropout, 
                                       batch_first=batch_first,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       input_size=input_size,
                                       num_frames=num_frames)
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        tgt: torch.Tensor,              # (B, T, C, H, W)
        memory: torch.Tensor,           # (B, S, C, H, W)
        memory_mask=None,
        memory_key_padding_mask=None
    ):
        batches, frames, channels, height, width = tgt.shape
        for i, layer in enumerate(self.layers):
            tgt = layer(tgt, memory, memory_mask, memory_key_padding_mask, first_layer=i==0)
        tgt = rearrange(tgt, 'b (f t) d -> b f t d', f=frames)
        return tgt


class SpatialCrossAttentionDecoder(nn.Module):
    def __init__(
        self,
        d_model=32,
        nhead=4,
        num_layers=4,
        dim_feedforward=128,
        dropout=0.1,
        patch_size=16,
        in_channels=3,
        input_size=96,
        num_frames=8,
        batch_first=True,
    ):
        """
        A stack of 'num_layers' SpatialCrossAttentionLayer layers.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            SpatialTemperalCrossAttentionLayer(d_model=d_model, 
                                       nhead=nhead, 
                                       dim_feedforward=dim_feedforward, 
                                       dropout=dropout, 
                                       batch_first=batch_first,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       input_size=input_size,
                                       num_frames=num_frames)
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        tgt: torch.Tensor,              # (B, T, C, H, W)
        memory: torch.Tensor,           # (B, S, C, H, W)
        memory_mask=None,
        memory_key_padding_mask=None
    ):
        batches, frames, channels, height, width = tgt.shape
        for i, layer in enumerate(self.layers):
            tgt = layer(tgt, memory, memory_mask, memory_key_padding_mask, first_layer=i==0)
        tgt = rearrange(tgt, 'b (f t) d -> b f t d', f=frames)
        return tgt

class AdaptiveSpatialTemperalCrossAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        input_size=96,
        patch_size=16,
        in_channels=3,
        num_frames=8,
        dropout: float = 0.1,
        batch_first: bool = True,  # new argument
    ):
        """
        A single Transformer Decoder layer with a spatial-attention and a cross-attention layer.
        Consists of:
         - Spatial attention sub-layer
         - Cross-attention sub-layer
         - Feed-forward sub-layer
         - Residual connections and LayerNorm in each sub-layer

        Args:
            d_model:         Dimension of embeddings.
            nhead:           Number of attention heads.
            dim_feedforward: Hidden layer size in the feed-forward network.
            dropout:         Dropout probability.
            batch_first:     If True, expects input shape (B, T, E).
                             If False, expects input shape (T, B, E).
        """
        super().__init__()
        # Spatial-Attention: query=decoder input, key/value=decoder input
        self.spatial_attn = TransformerBlock(
            hidden_size=d_model,
            num_heads=nhead,
            mlp_ratio=int(dim_feedforward / d_model),
        )
        self.temp_attn = TransformerBlock(
            hidden_size=d_model,
            num_heads=nhead,
            mlp_ratio=int(dim_feedforward / d_model),
        )
        # Cross-Attention: query=decoder input, key/value=memory (encoder output)
        # We pass batch_first=batch_first to align shapes accordingly:

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first
        )

        # LayerNorms for the two sub-layers (cross-attn + feed-forward)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network (simple 2-layer MLP)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

        self.batch_first = batch_first
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, d_model)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, d_model), requires_grad=False)
        temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], self.temp_embed.shape[-2])
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 8 * d_model, bias=True)
        )

    def forward(
        self,
        tgt: torch.Tensor,            # (batch_size, tgt_len, C, H, W) if batch_first=True
        memory: torch.Tensor,         # (batch_size, src_len, C, H, W) if batch_first=True
        c: torch.Tensor,  # Optional conditioning tensor
        memory_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
        first_layer: bool = False
    ):
        """
        Args:
            tgt:
                Decoder input embeddings. Shape depends on batch_first:
                  - (B, T, d_model) if batch_first=True
                  - (T, B, d_model) if batch_first=False
            memory:
                Encoder output / "memory". Shape depends on batch_first:
                  - (B, S, d_model) if batch_first=True
                  - (S, B, d_model) if batch_first=False
            memory_mask:
                Optional mask of shape (T, S) if batch_first=True, or (T, S) swapped if not.
            memory_key_padding_mask:
                Optional mask of shape (B, S) indicating padding in memory.

        Returns:
            out:
                Updated decoder embeddings, same shape as `tgt`.
        """
        batches = tgt.shape[0]
        if first_layer:
            tgt = rearrange(tgt, 'b f c h w -> (b f) c h w')
            tgt = self.x_embedder(tgt) + self.pos_embed
        else:
            tgt = rearrange(tgt, 'b (f t) d -> (b f) t d', t=self.pos_embed.shape[1])
        # ---- Spatial-Attention Sub-layer ----
        if len(memory.shape) == 4:
            memory = rearrange(memory, 'b f t d -> b (f t) d', b=batches)
        c_spatial = repeat(c, 'b d -> (b f) d', f=self.temp_embed.shape[1])
        tgt = self.spatial_attn(tgt, c_spatial)
        # ----- Temporal-Attention Sub-layer ----
        tgt = rearrange(tgt, '(b f) t d -> (b t) f d', b=batches)
        if first_layer:
            tgt = tgt + self.temp_embed 
        c_temp = repeat(c, 'b d -> (b t) d', t=self.pos_embed.shape[1])
        tgt = self.temp_attn(tgt, c_temp)
        # ---- Cross-Attention Sub-layer ----
        tgt = rearrange(tgt, '(b t) f d -> b (f t) d', b=batches)
        shift_msa, scale_msa, shift_ca, scale_ca, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(8, dim=1)
        # Q = tgt, K = memory, V = memory
        x_attn_out, _ = self.cross_attn(
            query=modulate(self.norm1(tgt), shift_msa, scale_msa),
            key=modulate(memory, shift_ca, scale_ca),
            value=modulate(memory, shift_ca, scale_ca),
            attn_mask=memory_mask,                 # shape depends on batch_first
            key_padding_mask=memory_key_padding_mask
        )
        # Residual + LayerNorm
        tgt = tgt + gate_msa.unsqueeze(1) * x_attn_out
        # ---- Feed-Forward Sub-layer ----
        ffn_out = self.ffn(modulate(self.norm2(tgt), shift_mlp, scale_mlp))
        out = tgt + gate_mlp.unsqueeze(1) * ffn_out
        return out

class AdaptiveSpatialCrossAttentionDecoder(nn.Module):
    def __init__(
        self,
        d_model=32,
        nhead=4,
        num_layers=4,
        dim_feedforward=128,
        dropout=0.1,
        patch_size=16,
        in_channels=3,
        input_size=96,
        num_frames=8,
        batch_first=True,
    ):
        """
        A stack of 'num_layers' SpatialCrossAttentionLayer layers.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            AdaptiveSpatialTemperalCrossAttentionLayer(d_model=d_model, 
                                       nhead=nhead, 
                                       dim_feedforward=dim_feedforward, 
                                       dropout=dropout, 
                                       batch_first=batch_first,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       input_size=input_size,
                                       num_frames=num_frames)
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        tgt: torch.Tensor,              # (B, T, C, H, W)
        memory: torch.Tensor,           # (B, S, C, H, W)
        c: torch.Tensor,  # Optional conditioning tensor
        memory_mask=None,
        memory_key_padding_mask=None
    ):
        batches, frames, channels, height, width = tgt.shape
        for i, layer in enumerate(self.layers):
            tgt = layer(tgt, memory, c, memory_mask, memory_key_padding_mask, first_layer=i==0)
        tgt = rearrange(tgt, 'b (f t) d -> b f t d', f=frames)
        return tgt

class CrossAttentionOnlyDecoder(nn.Module):
    def __init__(
        self,
        d_model=32,
        nhead=4,
        num_layers=4,
        dim_feedforward=128,
        dropout=0.1,
        batch_first=False,
    ):
        """
        A stack of 'num_layers' CrossAttentionLayer layers.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, nhead, dim_feedforward, dropout, batch_first)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        tgt: torch.Tensor,              # (B, T, d_model)
        memory: torch.Tensor,           # (B, S, d_model)
        memory_mask=None,
        memory_key_padding_mask=None
    ):
        x = tgt
        for layer in self.layers:
            x = layer(x, memory, memory_mask, memory_key_padding_mask)
        return x  # (B, T, d_model)

class VitMeanPooler(nn.Module):
    def forward(self, x):
        x = x.mean(dim=(-2, -1))
        return x

class ResNetPooler(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 512)
    
    def forward(self, x):
        # x shape: (batch_size, 512, 7, 7)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def collate_fn(batch):
    """
    Collate function for the dataloader (filter out None values)
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

# From pi-zero lerobot implementation
def get_safe_dtype(dtype: torch.dtype, device: str | torch.device):
    """
    mps is currently not compatible with float64
    """
    if isinstance(device, torch.device):
        device = device.type
    if device == "mps" and dtype == torch.float64:
        return torch.float32
    else:
        return dtype

def create_sinusoidal_pos_embedding_pi0(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb

def sample_points_on_polygon(coord, scale, sides, observation_height=96, observation_width=96):
    """
    Sample the vertices of a regular `sides`-gon (plus the center point).

    Args:
        coord: (x, y) center of the polygon
        scale: diameter (in your 512-based units) of the polygon
        sides: number of sides (e.g. 3=triangle, 4=square, 5=pentagon)

    Returns:
        xs, ys: lists of integer x and y coordinates of each vertex, then center
    """
    xs, ys = [], []
    for i in range(sides):
        angle = 2 * np.pi / sides * i
        x = int(int(coord[0] * observation_height / 512) + math.sin(angle) * scale * observation_height / 512)
        y = int(int(coord[1] * observation_width / 512) + math.cos(angle) * scale * observation_width / 512)
        xs.append(x)
        ys.append(y)

    # finally, include the center point
    xs.append(int(coord[0] * observation_height / 512))
    ys.append(int(coord[1] * observation_width / 512))
    return xs, ys

def render_action_on_image(images, sampled_actions, robot_zero_coord, color=(65, 105, 225)):
    """
    Render action points on images.

    Args:
        images: tensor of shape (N, H, W, C) where N is the number of images,
                H is the height, W is the width, and C is the number of channels.
        sampled_actions: tensor of shape (N, M, 2) where N is the number of images,
                       M is the number of action points, and 2 represents (x, y) coordinates.
        color: tuple of RGB values to render the action points.
    """
    robot_zero_coord = robot_zero_coord.to(images.device)  # Ensure robot_zero_coord is on the same device as images
    action_coords = sampled_actions / 512 * 96 + repeat(robot_zero_coord, 'n c -> b n c', b=images.shape[0])
    action_coords = action_coords.type(torch.long)  # Convert to int8 for indexing
    batch_indices = torch.arange(images.shape[0])[:, None].to(torch.long)  # (N, 1)
    x_coords = action_coords[:, :, 0].to(torch.long)  # (N, M)
    y_coords = action_coords[:, :, 1].to(torch.long)  # (N, M)
    x_coords = torch.clamp(x_coords, 0, images.shape[2] - 1)  # Ensure x coordinates are within bounds
    y_coords = torch.clamp(y_coords, 0, images.shape[1] - 1)  # Ensure y coordinates are within bounds
    images[batch_indices, y_coords, x_coords] = torch.tensor(color, dtype=images.dtype, device=images.device)  # Render action points
    return images

def render_action_on_image_np(images, robot_zero_coord, sampled_actions, color=(65, 105, 225)):
    """
    Render action points on images.

    Args:
        images: tensor of shape (N, H, W, C) where N is the number of images,
                H is the height, W is the width, and C is the number of channels.
        sampled_actions: tensor of shape (N, M, 2) where N is the number of images,
                       M is the number of action points, and 2 represents (x, y) coordinates.
        color: tuple of RGB values to render the action points.
    """
    
    action_coords = sampled_actions / 512 * 96 + np.tile(robot_zero_coord, (images.shape[0], 1, 1))
    action_coords = action_coords.astype(np.long)  # Convert to int8 for indexing
    batch_indices = np.arange(images.shape[0])[:, None]  # (N, 1)
    x_coords = action_coords[:, :, 0] # (N, M)
    y_coords = action_coords[:, :, 1] # (N, M)
    images[batch_indices, y_coords, x_coords] = np.array(color, dtype=np.long)  # Render action points
    return images

def position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100) -> torch.Tensor:
    """
    Convert 2D position grid (HxWx2) to sinusoidal embeddings (HxWxC)

    Args:
        pos_grid: Tensor of shape (H, W, 2) containing 2D coordinates
        embed_dim: Output channel dimension for embeddings

    Returns:
        Tensor of shape (H, W, embed_dim) with positional embeddings
    """
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)  # Flatten to (H*W, 2)

    # Process x and y coordinates separately
    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)  # [1, H*W, D/2]
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)  # [1, H*W, D/2]

    # Combine and reshape
    emb = torch.cat([emb_x, emb_y], dim=-1)  # [1, H*W, D]

    return emb.view(H, W, embed_dim)  # [H, W, D]


def make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    device = pos.device
    omega = torch.arange(embed_dim // 2, dtype=torch.float32 if device.type == "mps" else torch.double, device=device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.float()


# Inspired by https://github.com/microsoft/moge


def create_uv_grid(
    width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None
) -> torch.Tensor:
    """
    Create a normalized UV grid of shape (width, height, 2).

    The grid spans horizontally and vertically according to an aspect ratio,
    ensuring the top-left corner is at (-x_span, -y_span) and the bottom-right
    corner is at (x_span, y_span), normalized by the diagonal of the plane.

    Args:
        width (int): Number of points horizontally.
        height (int): Number of points vertically.
        aspect_ratio (float, optional): Width-to-height ratio. Defaults to width/height.
        dtype (torch.dtype, optional): Data type of the resulting tensor.
        device (torch.device, optional): Device on which the tensor is created.

    Returns:
        torch.Tensor: A (width, height, 2) tensor of UV coordinates.
    """
    # Derive aspect ratio if not explicitly provided
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    # Compute normalized spans for X and Y
    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    # Establish the linspace boundaries
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    # Generate 1D coordinates
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)

    # Create 2D meshgrid (width x height) and stack into UV
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    uv_grid = torch.stack((uu, vv), dim=-1)

    return uv_grid
