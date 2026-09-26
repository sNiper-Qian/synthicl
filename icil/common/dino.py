import torch
import torch.nn as nn
from contextlib import nullcontext
from einops import rearrange
from icil.common.lora import count_lora_parameters


class DINOv3FeatureExtractor(nn.Module):
    """DINOv3 feature extractor for semantic features.
    
    Efficient frozen feature extraction with optional multi-layer aggregation:
    - Pre-registered mean/std buffers (no tensor creation per forward)
    - Direct forward path with no branching
    - Uses HuggingFace transformers for gated model access
    - Multi-layer mode: concatenates features from multiple transformer layers
      for richer semantic + spatial representation (established best practice
      from DPT, Depth Anything V2, ViT-Adapter)
    """
    
    # Model configurations: model_name -> (hf_model_id, feature_dim, patch_size, num_layers)
    MODEL_CONFIGS = {
        'dinov3_vits16': ('facebook/dinov3-vits16-pretrain-lvd1689m', 384, 16, 12),
        'dinov3_vits16_plus': ('facebook/dinov3-vits16plus-pretrain-lvd1689m', 384, 16, 12),
    }
    
    # Default layer indices for multi-layer extraction (evenly spaced)
    # For 12-layer ViT-S: layers 2, 5, 8, 11 (0-indexed) capture low→high semantics
    DEFAULT_LAYER_INDICES = {
        12: [2, 5, 8, 11],      # ViT-S/B (12 layers)
        24: [5, 11, 17, 23],    # ViT-L (24 layers)
        40: [9, 19, 29, 39],    # ViT-g (40 layers)
    }
    
    def __init__(
        self,
        model_name: str = 'dinov3_vits16_plus',
        device: str = 'cuda',
        dino_repo: str = None,
        use_multi_layer: bool = False,
        layer_indices: list = None,
        use_lora: bool = False,
        lora_rank: int = 32,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
        lora_target_modules: list = None,
        lora_layers: list = None,
    ):
        """Initialize DINOv3 feature extractor.

        Args:
            model_name: Model name from MODEL_CONFIGS
            device: Device to load model on
            dino_repo: Ignored (kept for API compatibility)
            use_multi_layer: If True, extract and concatenate features from multiple
                transformer layers (4× feature dim). Recommended for dense prediction.
            layer_indices: Which layers to extract (0-indexed). Default: evenly spaced
                layers for the model architecture (e.g., [2,5,8,11] for 12-layer ViT-S).
            use_lora: If True, apply LoRA adaptation to attention layers
            lora_rank: LoRA rank (8-64, higher = more capacity)
            lora_alpha: LoRA alpha scaling (typically 16-32)
            lora_dropout: Dropout rate for LoRA layers
            lora_target_modules: Which projections to adapt ['q', 'k', 'v', 'o']
                Default: ['q', 'v'] (query and value projections)
            lora_layers: Which transformer layers to apply LoRA to (0-indexed)
                Default: last 4 layers [8,9,10,11] for 12-layer ViT-S
        """
        super().__init__()

        self.device = device
        self.model_name = model_name
        self.use_multi_layer = use_multi_layer
        self.use_lora = use_lora

        print(f"Loading DINOv3 model: {model_name}...")

        # Get model config
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODEL_CONFIGS.keys())}")

        hf_model_id, base_feature_dim, self.patch_size, num_layers = self.MODEL_CONFIGS[model_name]

        # Set layer indices for multi-layer extraction
        if use_multi_layer:
            self.layer_indices = layer_indices or self.DEFAULT_LAYER_INDICES.get(num_layers, [2, 5, 8, 11])
            self.feature_dim = base_feature_dim * len(self.layer_indices)
            print(f"Multi-layer mode: extracting from layers {self.layer_indices} → {self.feature_dim}D")
        else:
            self.layer_indices = None
            self.feature_dim = base_feature_dim

        self._base_feature_dim = base_feature_dim  # Store for reference
        self._num_layers = num_layers  # Store for LoRA

        # Load from HuggingFace using AutoModel for better compatibility
        try:
            from transformers import AutoModel
            print(f"Loading DINOv3 model from HuggingFace: {hf_model_id}")
            self.model = AutoModel.from_pretrained(hf_model_id)
            print("✓ Loaded DINOv3 with pretrained weights via HuggingFace")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DINOv3 model from HuggingFace ({hf_model_id}). "
                f"Make sure you have access to the model. Original error: {e}"
            )

        self.model.to(device)
        self.model.eval()

        # Apply LoRA if requested (before freezing)
        if use_lora:
            # print(f"Applying LoRA adaptation: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")
            from icil.common.lora import apply_lora_to_vit

            # Default: adapt last 4 layers (most task-specific)
            if lora_layers is None:
                lora_layers = list(range(num_layers - 4, num_layers))  # [8,9,10,11] for 12-layer

            # Default: adapt Q and V projections
            if lora_target_modules is None:
                lora_target_modules = ['q', 'v']

            self.model = apply_lora_to_vit(
                self.model,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=lora_target_modules,
                layer_indices=lora_layers,
            )

            # Unfreeze LoRA parameters
            from icil.common.lora import get_lora_parameters
            lora_params = get_lora_parameters(self.model)
            for param in lora_params:
                param.requires_grad = True

            # Enable gradient checkpointing to reduce memory (recompute activations
            # during backward instead of storing them — essential for 16GB GPUs)
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print(f"✓ Gradient checkpointing enabled for memory efficiency")

            print(f"✓ LoRA enabled: {len(lora_params)} parameters trainable")
        else:
            # Freeze all parameters (original behavior)
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Pre-register normalization tensors as buffers (efficient, moves with model)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))
        
        print(f"DINOv3 output feature dimension: {self.feature_dim}")
        print(f"Patch size: {self.patch_size}")

    def forward(self, images, return_separate_layers=False, chunk_size=64):
        """
        Extract dense DINOv3 features from images.

        Note: When LoRA is enabled, gradients will flow through this function.
        When LoRA is disabled, this runs in inference mode (no gradients).

        Args:
            images: (B, H, W, 3) batch of RGB image tensors OR (H, W, 3) single image in [0, 1] or [0, 255]
            chunk_size: Max images to process through DINO at once (prevents GPU OOM for large B*T)

        Returns:
            features: (B, C, H', W') batch of feature maps OR (C, H', W') for single image
                     C = base_dim if use_multi_layer=False, else base_dim * num_layers
        """
        # Handle single image case
        single_image = False
        # Put channel dimension on the last
        images = images.permute(0, 2, 3, 1) if images.ndim == 4 else images.permute(1, 2, 0)
        if images.ndim == 3:
            images = images.unsqueeze(0)
            single_image = True

        # Move to device
        images = images.to(self.device)

        B = images.shape[0]
        H, W = images.shape[1:3]
        H_new = (H // self.patch_size) * self.patch_size
        W_new = (W // self.patch_size) * self.patch_size

        # Permute to (B, 3, H, W)
        images_tensor = images.permute(0, 3, 1, 2)

        # Resize if needed
        if H != H_new or W != W_new:
            images_tensor = torch.nn.functional.interpolate(
                images_tensor, size=(H_new, W_new), mode='bilinear', align_corners=False
            )

        # Normalize: detect uint8 scale and convert
        if images_tensor.max() > 1.5:
            images_tensor = images_tensor / 255.0

        # images_tensor = (images_tensor - self.mean) / self.std

        # Get prefix token count (CLS + registers to skip)
        num_register_tokens = getattr(self.model.config, 'num_register_tokens', 0)
        num_prefix_tokens = 1 + num_register_tokens

        # Spatial dimensions for reshaping
        h = H_new // self.patch_size
        w = W_new // self.patch_size

        # Chunked processing to prevent GPU OOM for large batches (e.g. B*T > chunk_size)
        if B > chunk_size:
            return {"feature_map": self._extract_features_chunked(
                images_tensor, num_prefix_tokens, h, w,
                return_separate_layers, single_image, chunk_size
            )}

        # Skip autograd on frozen backbone (saves ~5% per forward pass)
        # When LoRA is active, gradients must flow through adapted layers
        grad_ctx = nullcontext() if self.use_lora else torch.no_grad()

        with grad_ctx:
            if self.use_multi_layer:
                # Multi-layer extraction: get hidden states from all layers
                outputs = self.model(pixel_values=images_tensor, output_hidden_states=True, return_dict=True)

                # hidden_states is tuple of (num_layers + 1) tensors, index 0 is embeddings
                # So layer i's output is at hidden_states[i + 1]
                multi_layer_features = []
                for layer_idx in self.layer_indices:
                    # +1 because hidden_states[0] is the input embeddings
                    layer_features = outputs.hidden_states[layer_idx + 1][:, num_prefix_tokens:, :]

                    # Reshape to (B, C, h, w)
                    layer_features = rearrange(layer_features, 'b (h w) c -> b c h w', h=h, w=w)
                    multi_layer_features.append(layer_features)

                if return_separate_layers:
                    return [f.clone() for f in multi_layer_features]

                # Concatenate along feature dimension
                patch_features = torch.cat(multi_layer_features, dim=1)
            else:
                # Single-layer extraction (original behavior)
                outputs = self.model(pixel_values=images_tensor, output_hidden_states=False, return_dict=True)
                patch_features = outputs.last_hidden_state[:, num_prefix_tokens:, :]
                # Reshape to (B, C, h, w)
                patch_features = rearrange(patch_features, 'b (h w) c -> b c h w', h=h, w=w)

        # Return single image features without batch dimension
        if single_image:
            return {"feature_map": patch_features.squeeze(0)}

        # return patch_features
        return {"feature_map": patch_features}

    def _extract_features_chunked(self, images_tensor, num_prefix_tokens, h, w,
                                   return_separate_layers, single_image, chunk_size):
        """Process images in chunks to avoid GPU OOM."""
        B = images_tensor.shape[0]
        all_features = []

        grad_ctx = nullcontext() if self.use_lora else torch.no_grad()

        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            chunk = images_tensor[start:end]

            with grad_ctx:
                if self.use_multi_layer:
                    outputs = self.model(pixel_values=chunk, output_hidden_states=True, return_dict=True)
                    chunk_layers = []
                    for layer_idx in self.layer_indices:
                        layer_features = outputs.hidden_states[layer_idx + 1][:, num_prefix_tokens:, :]
                        layer_features = rearrange(layer_features, 'b (h w) c -> b c h w', h=h, w=w)
                        chunk_layers.append(layer_features)

                    if return_separate_layers:
                        if not all_features:
                            all_features = [[] for _ in chunk_layers]
                        for i, lf in enumerate(chunk_layers):
                            all_features[i].append(lf.clone())
                        del outputs, chunk_layers
                        continue

                    chunk_feat = torch.cat(chunk_layers, dim=1)
                    del outputs, chunk_layers
                else:
                    outputs = self.model(pixel_values=chunk, output_hidden_states=False, return_dict=True)
                    chunk_feat = outputs.last_hidden_state[:, num_prefix_tokens:, :]
                    chunk_feat = rearrange(chunk_feat, 'b (h w) c -> b c h w', h=h, w=w)
                    del outputs

            all_features.append(chunk_feat)

        if return_separate_layers:
            return [torch.cat(layer_list, dim=0).clone() for layer_list in all_features]

        patch_features = torch.cat(all_features, dim=0)

        if single_image:
            return patch_features.squeeze(0)

        return patch_features



class DINOv2FeatureExtractor(nn.Module):
    """DINOv2 feature extractor for semantic features.
    
    Supports optional multi-layer aggregation for richer features:
    - Multi-layer mode: concatenates features from multiple transformer layers
      (established best practice from DPT, Depth Anything V2, ViT-Adapter)
    """
    
    # Model configurations: model_name -> (feature_dim, patch_size, num_layers)
    MODEL_CONFIGS = {
        'dinov2_vits14': (384, 14, 12),
        'dinov2_vitb14': (768, 14, 12),
        'dinov2_vitl14': (1024, 14, 24),
        'dinov2_vitg14': (1536, 14, 40),
    }
    
    # Default layer indices for multi-layer extraction (evenly spaced)
    DEFAULT_LAYER_INDICES = {
        12: [2, 5, 8, 11],      # ViT-S/B (12 layers)
        24: [5, 11, 17, 23],    # ViT-L (24 layers)
        40: [9, 19, 29, 39],    # ViT-g (40 layers)
    }

    def __init__(
        self, 
        model_name: str = 'dinov2_vits14', 
        device: str = 'cuda',
        use_multi_layer: bool = False,
        layer_indices: list = None,
    ):
        """Initialize DINOv2 feature extractor.
        
        Args:
            model_name: Model name (e.g., 'dinov2_vits14', 'dinov2_vitb14')
            device: Device to load model on
            use_multi_layer: If True, extract and concatenate features from multiple
                transformer layers (4× feature dim). Recommended for dense prediction.
            layer_indices: Which layers to extract (0-indexed). Default: evenly spaced.
        """
        super().__init__()

        self.device = device
        self.model_name = model_name
        self.use_multi_layer = use_multi_layer

        print(f"Loading DINOv2 model: {model_name}...")

        self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        self.model.to(device)
        self.model.eval()

        # Freeze DINOv2 parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Get model config
        if model_name in self.MODEL_CONFIGS:
            base_feature_dim, self.patch_size, num_layers = self.MODEL_CONFIGS[model_name]
        else:
            # Fallback for unknown models
            base_feature_dim, self.patch_size, num_layers = 384, 14, 12
        
        # Set layer indices for multi-layer extraction
        if use_multi_layer:
            self.layer_indices = layer_indices or self.DEFAULT_LAYER_INDICES.get(num_layers, [2, 5, 8, 11])
            self.feature_dim = base_feature_dim * len(self.layer_indices)
            print(f"Multi-layer mode: extracting from layers {self.layer_indices} → {self.feature_dim}D")
        else:
            self.layer_indices = None
            self.feature_dim = base_feature_dim
        
        self._base_feature_dim = base_feature_dim
        self._num_layers = num_layers

        print(f"DINOv2 output feature dimension: {self.feature_dim}")
        print(f"Patch size: {self.patch_size}")

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    @torch.inference_mode()
    def extract_features(self, images, return_separate_layers=False):
        """
        Extract dense DINOv2 features from images.

        Args:
            images: (B, H, W, 3) batch of RGB image tensors OR (H, W, 3) single image in [0, 1] or [0, 255]

        Returns:
            features: (B, C, H', W') batch of feature maps OR (C, H', W') for single image
                     C = base_dim if use_multi_layer=False, else base_dim * num_layers
        """
        # Handle single image case
        single_image = False
        if images.ndim == 3:
            images = images.unsqueeze(0)
            single_image = True

        # Move to device
        images = images.to(self.device)

        B = images.shape[0]
        H, W = images.shape[1:3]
        H_new = (H // self.patch_size) * self.patch_size
        W_new = (W // self.patch_size) * self.patch_size

        # Permute to (B, 3, H, W)
        images_tensor = images.permute(0, 3, 1, 2)

        # Resize if needed
        if H != H_new or W != W_new:
            images_tensor = torch.nn.functional.interpolate(
                images_tensor, size=(H_new, W_new), mode='bilinear', align_corners=False
            )

        # Normalize
        if images_tensor.max() > 1.5:  # detect uint8 scale
            images_tensor = images_tensor / 255.0

        images_tensor = (images_tensor - self.mean) / self.std

        # Spatial dimensions for reshaping
        h = H_new // self.patch_size
        w = W_new // self.patch_size

        if self.use_multi_layer:
            # Multi-layer extraction: use get_intermediate_layers API
            # DINOv2 provides this convenient method for extracting from multiple layers
            intermediate_outputs = self.model.get_intermediate_layers(
                images_tensor, 
                n=self.layer_indices,  # List of layer indices
                reshape=True,  # Return as spatial maps (B, C, h, w)
                return_class_token=False,  # Skip CLS token
                norm=True,  # Apply layer norm
            )
            
            if return_separate_layers:
                # Clone items in list to exit @inference_mode with regular Tensors
                return [f.clone() for f in intermediate_outputs]
                
            # Concatenate along feature dimension
            patch_features = torch.cat(intermediate_outputs, dim=1)
        else:
            # Single-layer extraction (original behavior)
            features = self.model.forward_features(images_tensor)
            patch_features = features["x_norm_patchtokens"]
            # Reshape to spatial maps: (B, num_patches, C) -> (B, C, h, w)
            patch_features = rearrange(patch_features, 'b (h w) c -> b c h w', h=h, w=w)

        # Return single image features without batch dimension
        if single_image:
            return patch_features.squeeze(0)

        return patch_features

if __name__ == "__main__":
    # Quick test to verify DINOv3 feature extractor works
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = DINOv3FeatureExtractor(model_name='dinov3_vits16_plus', device=device, use_multi_layer=False, lora_rank=32,
                                        use_lora=True,
                                        lora_alpha=32.0,
                                        lora_dropout=0.0,
                                        lora_target_modules=["q", "v"],      # adapt query/value projections
                                        lora_layers=[8, 9, 10, 11],          # last 4 transformer blocks
                                        )
    # Count LoRA parameters
    if extractor.use_lora:
        num_lora_params = count_lora_parameters(extractor.model)
        print(f"Number of trainable LoRA parameters: {num_lora_params}")
    # Count total parameters
    total_params = sum(p.numel() for p in extractor.model.parameters())
    print(f"Total model parameters: {total_params}")
                                        
    # Create dummy image batch (B=2, H=224, W=224, 3 channels)
    dummy_images = torch.rand(2, 3, 224, 224)  # In [0, 1]
    
    features = extractor(dummy_images)["feature_map"]
    print(f"Extracted features shape: {features.shape}")