import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
from icil.config.policy_config import PolicyConfig
from timm.models.vision_transformer import Mlp, PatchEmbed
# from vggt.models.vggt import VGGT
from typing import List, Tuple, Union
import math
from torch import Tensor
import einops
from icil.common.utils import create_uv_grid, position_grid_to_embed

class ResNetBackbone(nn.Module):
    def __init__(self, config: PolicyConfig=None):
        super().__init__()
        self.backbone = getattr(torchvision.models, "resnet18")(
                    replace_stride_with_dilation=[False, False, False],
                    weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
                    norm_layer=FrozenBatchNorm2d,
                )
        print(f"Using ResNet18 backbone")
        self.body = IntermediateLayerGetter(self.backbone, return_layers={"layer4": "feature_map"})
        for param in self.body.parameters():
            param.requires_grad = True
        # unfreeze layer 4
        for param in self.body.layer4.parameters():
            param.requires_grad = True
            # print(f"Unfreezing layer 4 parameters: {param.shape}")
        # unfreeze layer 3
        for param in self.body.layer3.parameters():
            param.requires_grad = True
            # print(f"Unfreezing layer 3 parameters: {param.shape}")

    @torch.no_grad()
    def forward(self, x):
        return self.body(x)

class DINOv2BackBone(nn.Module):
    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.body = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.body.eval()
        for param in self.body.parameters():
            param.requires_grad = False
        self.num_channels = 384
        self.n_patches = (config.image_size[0] // 14, config.image_size[1] // 14)
    
    def forward(self, tensor):
        # `no_grad` keeps the frozen backbone out of autograd while returning a
        # regular tensor that trainable projection layers can consume.
        with torch.no_grad():
            xs = self.body.forward_features(tensor)["x_norm_patchtokens"]
            od = OrderedDict()
            od["0"] = xs.reshape(xs.shape[0], self.n_patches[0], self.n_patches[1], 384).permute(0, 3, 2, 1)
            # return od
            return {"feature_map": od["0"]}

class DINOv3BackBone(nn.Module):
    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        if not config.dino_repo or not config.dino_weights:
            raise ValueError("DINOv3 requires both dino_repo and dino_weights paths")
        self.body = torch.hub.load(
            config.dino_repo,
            "dinov3_vits16plus",
            source="local",
            weights=config.dino_weights,
        )
        self.body.eval()
        for param in self.body.parameters():
            param.requires_grad = False
        self.num_channels = 384
        self.n_patches = (config.image_size[0] // 16, config.image_size[1] // 16)
    
    def forward(self, tensor):
        # Do not use `inference_mode` here: its tensors cannot be saved by the
        # trainable feature projection during backward.
        with torch.no_grad():
            xs = self.body.forward_features(tensor)["x_prenorm"]
            if xs.shape[1] != self.n_patches[0] * self.n_patches[1]:
                xs = xs[:, -self.n_patches[0] * self.n_patches[1]:, :]   # patches are typically last
            od = OrderedDict()
            od["0"] = xs.reshape(xs.shape[0], self.n_patches[0], self.n_patches[1], 384).permute(0, 3, 2, 1)
            # return od
            return {"feature_map": od["0"]}

class AttentionPooler(nn.Module):
    def __init__(self, in_dim, out_dim, num_queries, num_heads=8, dropout=0.0):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, out_dim, bias=True)
        self.query = nn.Parameter(torch.randn(1, num_queries, out_dim))  # [1, 1, C]
        self.attn = nn.MultiheadAttention(embed_dim=out_dim, num_heads=num_heads,
                                          batch_first=True, dropout=dropout)
        self.ln = nn.LayerNorm(out_dim)
 
    def forward(self, x, key_padding_mask=None):
        """
        x: [B, N, in_dim]  (N = T*H*W)
        returns: [B, out_dim]
        """
        x = self.in_proj(x)                         # [B, N, out_dim]
        B = x.size(0)
        q = self.query.expand(B, -1, -1)            # [B, n_queries, out_dim]
        y, _ = self.attn(q, self.ln(x), self.ln(x),
                         key_padding_mask=key_padding_mask)
        return y                            # [B, n_queries, out_dim]

class Patchifier(nn.Module):
    def __init__(self, image_size, patch_size, dim_model):
        super().__init__()
        self.patchifier = PatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=dim_model,
            bias=True,
            flatten=False,
        )
    def forward(self, x):
        x = self.patchifier(x)
        return {"feature_map": x}
    
class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, num_convs):
        layers = []
        for i in range(num_convs):
            layers += [
                nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False
                ),
                nn.ReLU(inplace=True)
            ]
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        super().__init__(*layers)

class VGGTBackbone(nn.Module):
    def __init__(self,):
        super().__init__()
        self.aggregator = VGGT.from_pretrained("facebook/VGGT-1B").aggregator
        # Freeze the aggregator parameters
        for param in self.aggregator.parameters():
            param.requires_grad = False
        
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Implementation of the forward pass through the DPT head.

        This method processes a specific chunk of frames from the sequence.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.
            frames_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Feature maps or (predictions, confidence).
        """
        aggregated_tokens_list, ps_idx = self.aggregator(x)
        out = []
        dpt_idx = 0
        B, S, N, D = aggregated_tokens_list[0].shape
        patch_h, patch_w = 16, 16
        out = aggregated_tokens_list[-1][:, :, ps_idx:]
        out = einops.rearrange(out, "b s (h w) d -> (b s) d h w", h=patch_h, w=patch_w)
        return {"feature_map": out.view(B, S, *out.shape[1:]),}
    
    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., use_lora=False, attention_mode='math'):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attention_mode = attention_mode
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value, key_padding_mask=None, batch_first=False, return_averaged_attn=False):
        if not batch_first:
            query = query.transpose(0, 1)  # [B, N, C]
            key = key.transpose(0, 1)      # [B, N, C]
            value = value.transpose(0, 1)  # [B, N, C]
        
        if key_padding_mask is not None:
            query_mask = key_padding_mask.unsqueeze(-1)  # [B, N, 1]
            key_mask   = key_padding_mask.unsqueeze(1)   # [B, 1, N]
            attn_mask_2d = query_mask | key_mask  # [B, N, N]
        B, Nq, Cq = query.shape
        Bk, Nk, Ck = key.shape

        # Projections
        q = self.q_proj(query)   # [B, Nq, Cq]
        k = self.k_proj(key) # [B, Nk, Cq]
        v = self.v_proj(value) # [B, Nk, Cq]

        # Shape to multi-head: [B, heads, N*, head_dim]
        q = q.view(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        
        if self.attention_mode == 'xformers': # cause loss nan while using with amp
            # https://github.com/facebookresearch/xformers/blob/e8bd8f932c2f48e3a3171d06749eecbbf1de420c/xformers/ops/fmha/__init__.py#L135
            q_xf = q.transpose(1,2).contiguous()
            k_xf = k.transpose(1,2).contiguous()
            v_xf = v.transpose(1,2).contiguous()
            attn_bias = torch.zeros(
                                (B, 1, N, N),
                                device=x.device,
                                dtype=x.dtype
                            )
            if key_padding_mask is not None:
                attn_bias = attn_bias.masked_fill(attn_mask_2d.unsqueeze(1), float('-inf'))
            x = xformers.ops.memory_efficient_attention(q_xf, k_xf, v_xf, attn_bias=attn_bias).reshape(B, Nq, Cq)

        elif self.attention_mode == 'flash':
            # cause loss nan while using with amp
            # Let PyTorch select the supported SDPA kernel for this device.
            if key_padding_mask is not None:
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask_2d).reshape(B, Nq, Cq)
            else:
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v).reshape(B, Nq, Cq) # require pytorch 2.0

        elif self.attention_mode == 'math':
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if key_padding_mask is not None:
                attn = attn.masked_fill(attn_mask_2d, float('-inf'))
            attn = attn.softmax(dim=-1)
            # attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, Nq, Cq)

        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        if not batch_first:
            x = x.transpose(0, 1)  # [N, B, C]
        if return_averaged_attn:
            return x, attn.mean(dim=1)  # average over heads
        return x

class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed

class DinoVGGTFusionBackbone(nn.Module):
    """
    Backbone that combines DINOv2 and VGGT features via a single cross-attention layer.
    DINO tokens act as queries while VGGT tokens provide the keys and values.
    """

    def __init__(
        self,
        config: PolicyConfig,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dino = DINOv2BackBone(config)
        self.vggt = VGGTBackbone()

        self.dino_channels = self.dino.num_channels
        self.vggt_channels = 2048  # VGGT aggregator output channels.
        self.fusion_dim = config.dim_model

        self.query_proj = nn.Linear(self.dino_channels, self.fusion_dim)
        self.key_proj = nn.Linear(self.vggt_channels, self.fusion_dim)
        self.value_proj = nn.Linear(self.vggt_channels, self.fusion_dim)
        self.cross_attn = Attention(
            dim=self.fusion_dim,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout,
            attention_mode='flash',
        )
        self.dino_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(self.dino_channels//2)
        self.vggt_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(self.vggt_channels//2)
        # self.output_proj = nn.Linear(self.fusion_dim, self.dino_channels)
        self.num_channels = self.dino_channels

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: Tensor shaped [B, C, H, W] or [B, S, C, H, W] (S = sequence/camera length).

        Returns:
            Dict with fused feature map shaped like the DINO output.
        """
        squeeze_seq = False
        if x.ndim == 4:
            x = x.unsqueeze(1)
            squeeze_seq = True
        if x.ndim != 5:
            raise ValueError(f"Expected 4D or 5D tensor, but received tensor with shape {tuple(x.shape)}")

        b, s, c, h, w = x.shape
        dino_inp = x.reshape(b * s, c, h, w)
        with torch.no_grad():
            dino_feats = self.dino(dino_inp)["feature_map"]
        dino_feats = dino_feats + self.dino_feat_pos_embed(dino_feats)
        dino_feats = dino_feats.view(b, s, *dino_feats.shape[1:])
        dino_h, dino_w = dino_feats.shape[-2], dino_feats.shape[-1]
        dino_tokens = einops.rearrange(dino_feats, "b s c hh ww -> (b s) (hh ww) c")
        vggt_inp = x.reshape(b * s, 1, c, h, w)
        with torch.no_grad():
            vggt_feats = self.vggt(vggt_inp)["feature_map"]
        vggt_feats = vggt_feats + self.vggt_feat_pos_embed(vggt_feats.view(b * s, *vggt_feats.shape[2:]))
        vggt_tokens = einops.rearrange(vggt_feats, "b s c hh ww -> (b s) (hh ww) c")
        queries = self.query_proj(dino_tokens)
        keys = self.key_proj(vggt_tokens)
        values = self.value_proj(vggt_tokens)

        fused_tokens = self.cross_attn(queries, keys, values, batch_first=True)
        # fused_tokens = self.output_proj(fused_tokens + queries)

        fused = einops.rearrange(
            fused_tokens,
            "(b s) (hh ww) c -> b s c hh ww",
            b=b,
            s=s,
            hh=dino_h,
            ww=dino_w,
        )
        if squeeze_seq:
            fused = fused[:, 0]
        return {"feature_map": fused}

if __name__ == "__main__":
    config = PolicyConfig()
    # model = ResNetBackbone(config)
    # x = torch.randn(1, 3, 224, 224)
    # out = model(x)
    # print(out["feature_map"].shape)
    # encoder_img_feat_input_proj = nn.Conv2d(
    #                 512, 30, kernel_size=1
    #             )
    # out["feature_map"] = encoder_img_feat_input_proj(out["feature_map"])
    # print(out["feature_map"][:, :, None, :].shape)
    # model = DINOv2BackBone()
    model_config = PolicyConfig(n_encoder_layers=8, 
                                dim_model=512, 
                                dim_feedforward=2048, 
                                n_heads=8, 
                                dropout=0.1, 
                                pre_norm=False, 
                                pooling_strategy="none", 
                                patch_size=14, 
                                decoder_type="cross_attention", 
                                vision_backbone="dino_v2", 
                                image_size=(224, 224), 
                                action_channels=4,
                                obs_dim=7,
                                action_dim=7,
                                )
    model = DINOv3BackBone(model_config).to("cuda")
    x = torch.randn(3, 3, 224, 224).to("cuda")
    import time
    t1 = time.time()
    out = model(x)["feature_map"]
    t2 = time.time()
    print(f"Forward time: {t2 - t1:.4f} seconds")

    # pooler = ResNetPooler()
    # out = pooler(out)

    print(out.shape)
    # model = CNNBackbone()
    # inp = torch.randn(1, 3, 224, 224)
    # out = model(inp)
    # print(out.shape)  # torch.Size([1, 512, 7, 7])
