"""
LoRA (Low-Rank Adaptation) for Vision Transformers.

Implements parameter-efficient fine-tuning by adding low-rank decomposition
matrices to attention layers. Based on:
- LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
- SOTA 2024-2026 practices for adapting DINO/DINOv2/DINOv3

Key features:
- Backward compatible: Can be added to frozen models without breaking checkpoints
- Efficient: Only adds ~2M params for rank=32 on 4 layers
- Flexible: Can target Q, K, V, or output projections
"""

import torch
import torch.nn as nn
import math
from typing import List, Optional


class LoRALayer(nn.Module):
    """
    LoRA layer that wraps a linear layer with low-rank adaptation.

    Implements: h = W_0 x + (B A) x * (alpha / r)
    where:
    - W_0: frozen pretrained weights
    - A: (r, in_features) trainable down-projection
    - B: (out_features, r) trainable up-projection
    - r: rank (typically 8-64)
    - alpha: scaling factor (typically 16-32)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        alpha: float = 32.0,
        dropout: float = 0.0,
        device: str = 'cuda',
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA low-rank matrices (create on correct device)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device))

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize A with Kaiming uniform, B with zeros (standard LoRA init)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA adaptation.

        Args:
            x: Input tensor (*, in_features)

        Returns:
            LoRA output: (B A) x * scaling
        """
        # x @ A^T -> (*, rank)
        # result @ B^T -> (*, out_features)
        result = x @ self.lora_A.T
        result = self.dropout(result)
        result = result @ self.lora_B.T
        return result * self.scaling


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.

    Wraps an existing nn.Linear layer and adds LoRA on top.
    The original layer remains frozen.
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 32,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = linear
        # Get device from the linear layer
        device = next(linear.parameters()).device
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            device=device,
        )

        # Freeze original linear layer
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: frozen linear + LoRA adaptation."""
        return self.linear(x) + self.lora(x)


def apply_lora_to_vit(
    model: nn.Module,
    rank: int = 32,
    alpha: float = 32.0,
    dropout: float = 0.0,
    target_modules: List[str] = ['q', 'v'],
    layer_indices: Optional[List[int]] = None,
) -> nn.Module:
    """
    Apply LoRA to a Vision Transformer model (DINOv2/DINOv3).

    Args:
        model: The ViT model (e.g., DINOv3FeatureExtractor.model)
        rank: LoRA rank (8-64, higher = more capacity)
        alpha: LoRA alpha scaling (typically 16-32)
        dropout: Dropout rate for LoRA layers (0.0 = no dropout)
        target_modules: Which attention projections to adapt
            - 'q': query projection
            - 'k': key projection
            - 'v': value projection
            - 'o': output projection
        layer_indices: Which transformer layers to apply LoRA to.
            If None, applies to all layers. Recommended: last 4 layers [8,9,10,11]
            for 12-layer ViT-S (most task-specific).

    Returns:
        Modified model with LoRA layers inserted

    Example:
        >>> dino = DINOv3FeatureExtractor('dinov3_vits16_plus')
        >>> dino.model = apply_lora_to_vit(
        ...     dino.model,
        ...     rank=32,
        ...     alpha=32,
        ...     target_modules=['q', 'v'],
        ...     layer_indices=[8, 9, 10, 11]  # Last 4 layers
        ... )
    """
    # Map target module names to attribute names
    module_mapping = {
        'q': 'q_proj',
        'k': 'k_proj',
        'v': 'v_proj',
        'o': 'o_proj',
    }

    num_lora_layers = 0
    num_lora_params = 0

    # Get encoder layers (handle both standard ViT and DINOv3 structures)
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        # Standard HuggingFace ViT
        layers = model.encoder.layer
    elif hasattr(model, 'layer'):
        # DINOv3 structure
        layers = model.layer
    else:
        raise ValueError("Model structure not recognized. Expected model.encoder.layer or model.layer")

    # Determine which layers to modify
    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    print(f"\nApplying LoRA to Vision Transformer:")
    print(f"  Rank: {rank}, Alpha: {alpha}, Dropout: {dropout}")
    print(f"  Target modules: {target_modules}")
    print(f"  Target layers: {layer_indices} (out of {len(layers)} total)")

    for layer_idx in layer_indices:
        if layer_idx >= len(layers):
            print(f"  Warning: Layer {layer_idx} does not exist (max: {len(layers)-1}), skipping")
            continue

        layer = layers[layer_idx]

        # Get attention module (handle both structures)
        if hasattr(layer, 'attention'):
            # DINOv3 structure: layer.attention.{q,k,v,o}_proj
            attention = layer.attention
        elif hasattr(layer, 'self_attn'):
            # Some ViT variants
            attention = layer.self_attn
        else:
            print(f"  Warning: Layer {layer_idx} has no attention module, skipping")
            continue

        for target in target_modules:
            if target not in module_mapping:
                print(f"  Warning: Unknown target module '{target}', skipping")
                continue

            attr_name = module_mapping[target]

            # Check if module exists
            if not hasattr(attention, attr_name):
                print(f"  Warning: Layer {layer_idx} has no attribute '{attr_name}', skipping")
                continue

            # Get the original linear layer
            original_linear = getattr(attention, attr_name)

            if not isinstance(original_linear, nn.Linear):
                print(f"  Warning: Layer {layer_idx}.{attr_name} is not nn.Linear, skipping")
                continue

            # Wrap with LoRA
            lora_linear = LoRALinear(
                original_linear,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )

            # Replace the module
            setattr(attention, attr_name, lora_linear)

            num_lora_layers += 1
            # Count LoRA parameters: A (rank × in_features) + B (out_features × rank)
            num_lora_params += rank * (original_linear.in_features + original_linear.out_features)

    print(f"  ✓ Applied LoRA to {num_lora_layers} modules")
    print(f"  ✓ Added {num_lora_params:,} trainable parameters ({num_lora_params/1e6:.2f}M)")

    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all LoRA parameters from a model.

    Useful for creating separate optimizer groups for LoRA vs other parameters.

    Args:
        model: Model with LoRA layers

    Returns:
        List of LoRA parameters
    """
    lora_params = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_params.extend(module.parameters())
    return lora_params


def count_lora_parameters(model: nn.Module) -> int:
    """Count total number of LoRA parameters in model."""
    return sum(p.numel() for p in get_lora_parameters(model))


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA weights into the base model and remove LoRA modules.

    Computes W_merged = W_0 + (B @ A) * (alpha/rank) for each LoRA layer,
    then replaces LoRALinear with the plain nn.Linear. After merging, the
    model produces identical outputs but without any LoRA overhead.

    Args:
        model: Model with LoRA layers

    Returns:
        Model with merged weights (LoRA modules removed)
    """
    # Collect (parent, attr_name, merged_linear) so we can swap after iteration
    replacements = []
    for name, module in model.named_modules():
        for attr_name, child in module.named_children():
            if isinstance(child, LoRALinear):
                with torch.no_grad():
                    lora_weight = child.lora.lora_B @ child.lora.lora_A * child.lora.scaling
                    child.linear.weight.data += lora_weight
                replacements.append((module, attr_name, child.linear))

    for parent, attr_name, merged_linear in replacements:
        setattr(parent, attr_name, merged_linear)

    print(f"✓ Merged and removed {len(replacements)} LoRA modules")
    return model


def save_lora_checkpoint(model: nn.Module, path: str):
    """
    Save only LoRA parameters (not the full model).

    This is much more efficient than saving the entire model.

    Args:
        model: Model with LoRA layers
        path: Path to save checkpoint
    """
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_state_dict[name] = module.state_dict()

    torch.save(lora_state_dict, path)
    print(f"✓ Saved LoRA checkpoint to {path}")


def load_lora_checkpoint(model: nn.Module, path: str):
    """
    Load LoRA parameters from checkpoint.

    Args:
        model: Model with LoRA layers
        path: Path to checkpoint
    """
    lora_state_dict = torch.load(path)

    for name, module in model.named_modules():
        if isinstance(module, LoRALayer) and name in lora_state_dict:
            module.load_state_dict(lora_state_dict[name])

    print(f"✓ Loaded LoRA checkpoint from {path}")
