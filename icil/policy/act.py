import math
import random
from itertools import chain
from typing import Optional
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import einops
from torch import Tensor
from torchvision.ops import FrozenBatchNorm2d
from torchvision.models._utils import IntermediateLayerGetter
from icil.config.configuration_act import ACTConfig
from icil.common.utils import get_activation_fn, create_sinusoidal_pos_embedding_pi0
from icil.common.vision import DINOv2BackBone, Patchifier, ResNetBackbone, VGGTBackbone, DinoVGGTFusionBackbone, DINOv3BackBone
from icil.common.resnet_film import ResNetFilmBackbone
from icil.common.utils import SpatialCrossAttentionDecoder, AdaptiveSpatialCrossAttentionDecoder, SpatialTemperalSelfAttentionEncoder
from icil.policy.latte import (
    FinalLayer,
    PatchEmbed,
    TimestepEmbedder,
    UnconditionalFinalLayer,
    get_1d_sincos_temp_embed,
    get_2d_sincos_pos_embed,
    modulate,
)

class ACT(nn.Module):
    """Action Chunking Transformer: The underlying neural network for ACTPolicy.

    Note: In this code we use the terms `vae_encoder`, 'encoder', `decoder`. The meanings are as follows.
        - The `vae_encoder` is, as per the literature around variational auto-encoders (VAE), the part of the
          model that encodes the target data (a sequence of actions), and the condition (the robot
          joint-space).
        - A transformer with an `encoder` (not the VAE encoder) and `decoder` (not the VAE decoder) with
          cross-attention is used as the VAE decoder. For these terms, we drop the `vae_` prefix because we
          have an option to train this model without the variational objective (in which case we drop the
          `vae_encoder` altogether, and nothing about this model has anything to do with a VAE).

                                 Transformer
                                 Used alone for inference
                                 (acts as VAE decoder
                                  during training)
                                ┌───────────────────────┐
                                │             Outputs   │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                inputs    └─────┼──┘  │ image emb.      │
                                │    state emb.         │
                                └───────────────────────┘
    """

    def __init__(self, config: ACTConfig, vision_backbone: Optional[nn.Module] = None):
        super().__init__()
        self.config = config
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        self.use_robot_state = "observation.state" in config.input_shapes
        self.use_images = any(k.startswith("observation.image") for k in config.input_shapes)
        self.use_env_state = "observation.environment_state" in config.input_shapes
        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            # Projection layer for joint-space configuration to hidden dimension.
            if self.use_robot_state:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    config.input_shapes["observation.state"][0], config.dim_model
                )
            # Projection layer for action (joint-space target) to hidden dimension.
            self.vae_encoder_action_input_proj = nn.Linear(
                config.output_shapes["action"][0], config.dim_model
            )
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            # Fixed sinusoidal positional embedding for the input to the VAE encoder. Unsqueeze for batch
            # dimension.
            num_input_token_encoder = 1 + config.chunk_size
            if self.use_robot_state:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # Backbone for image feature extraction.
        if self.use_images:
            if config.use_film:
                if config.vision_backbone == "resnet18":
                    film_config = {
                        'use': True,
                        'use_in_layers': [1, 2, 3],
                        'task_embedding_dim': config.film_embedding_dim,
                        'film_planes': [64, 128, 256, 512],
                    }
                    self.backbone = ResNetFilmBackbone(
                        'resnet18_film', film_config=film_config
                    )
                else:
                    raise ValueError(
                        f"ResNet-Film backbone is only available for resnet18 backbone. Got {config.vision_backbone}."
                    )
            else:
                if config.vision_backbone == "dino_v2":
                    if vision_backbone is not None:
                        self.backbone = vision_backbone
                    else:
                        self.backbone = DINOv2BackBone()
                    for param in self.backbone.parameters():
                        param.requires_grad = False
                elif config.vision_backbone == "dino_v3":
                    if vision_backbone is not None:
                        self.backbone = vision_backbone
                    else:
                        self.backbone = DINOv3BackBone(config)
                    for param in self.backbone.parameters():
                        param.requires_grad = False
                elif config.vision_backbone == "dino_v3_lora":
                    if vision_backbone is not None:
                        self.backbone = vision_backbone
                    else:
                        self.backbone = DINOv3FeatureExtractor(model_name='dinov3_vits16_plus', device="cuda", use_multi_layer=False, lora_rank=32, use_lora=True)
                    # We unfreeze the backbone's parameters because with LoRA we are fine-tuning the entire backbone.
                elif config.vision_backbone == "resnet18":
                    if vision_backbone is not None:
                        self.backbone = vision_backbone
                    else:
                        backbone_model = getattr(torchvision.models, config.vision_backbone)(
                            replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                            weights=config.pretrained_backbone_weights,
                            norm_layer=FrozenBatchNorm2d,
                        )
                        # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
                        # feature map).
                        # Note: The forward method of this returns a dict: {"feature_map": output}.
                        for param in backbone_model.parameters():
                            param.requires_grad = False
                        self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
                    # self.backbone = TinyCNNBackbone()
                elif config.vision_backbone == "vggt":
                    if vision_backbone is not None:
                        self.backbone = vision_backbone
                    else:
                        self.backbone = DinoVGGTFusionBackbone()
                elif config.vision_backbone is None:
                    # patchify the image
                    self.backbone = Patchifier(config.image_size[0], config.patch_size, config.dim_model)

        # Transformer (acts as VAE decoder when training with the variational objective).
        if self.config.use_cross_attention and not self.config.use_spatial_temporal_encoder:
            self.encoder = ACTCrossAttentionEncoder(config)
        elif self.config.use_spatial_temporal_encoder and self.config.use_cross_attention:
            self.encoder = SpatialCrossAttentionDecoder(
                                d_model=config.dim_model,
                                nhead=config.n_heads,
                                num_layers=config.n_encoder_layers,
                                dim_feedforward=config.dim_feedforward,
                                dropout=config.dropout,
                                patch_size=1,
                                batch_first=True,
                                num_frames=config.num_frames,
                                in_channels=config.dim_model,  
                                input_size=config.image_size[0] // config.patch_size,
                            )
        else:
            self.encoder = ACTEncoder(config)

        if self.config.use_flow_as_auxiliary or self.config.use_image_as_auxiliary:
            self.action_decoder = ACTDecoder(config)
            if self.config.use_flow_as_auxiliary and self.config.use_adaLN and (self.config.use_flow_matching or self.config.use_diffusion):
                self.flow_decoder = ACTAdaptiveFlowDecoder(config)
            elif self.config.use_flow_as_auxiliary:
                self.flow_decoder = ACTFlowDecoder(config)
            if self.config.use_image_as_auxiliary:
                self.image_decoder = ACTImageDecoder(config)
        elif not self.config.use_flow:
            self.decoder = ACTDecoder(config)
        else:
            self.decoder = ACTFlowDecoder(config)
        
        if self.config.use_detection_as_auxiliary:
            self.detection_decoder = ACTDetectionDecoder(config)
            self.detection_decoder_pos_embed = nn.Embedding(1, config.dim_model)
            self.detection_head = nn.Linear(config.dim_model, 3)

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, (robot_state), (env_state), (image_feature_map_pixels)].
        if self.use_robot_state:
            self.encoder_robot_state_input_proj = nn.Linear(
                config.input_shapes["observation.state"][0], config.dim_model
            )
        if self.use_env_state:
            self.encoder_env_state_input_proj = nn.Linear(
                config.input_shapes["observation.environment_state"][0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.use_images:
            if config.vision_backbone == "dino_v2" or config.vision_backbone == "dino_v3":
                self.encoder_img_feat_input_proj = nn.Conv2d(
                    self.backbone.num_channels, config.dim_model, kernel_size=1
                )
            elif config.vision_backbone == "dino_v3_lora":
                self.encoder_img_feat_input_proj = nn.Conv2d(
                    self.backbone.feature_dim, config.dim_model, kernel_size=1
                )
            elif config.vision_backbone == "resnet18":
                if not config.use_film:
                    self.encoder_img_feat_input_proj = nn.Conv2d(
                        512, config.dim_model, kernel_size=1
                    )
                else:
                    self.encoder_img_feat_input_proj = nn.Conv2d(
                        512, config.dim_model, kernel_size=1
                    )
            elif config.vision_backbone == "vggt":
                self.encoder_img_feat_input_proj = nn.Conv2d(
                    512, config.dim_model, kernel_size=1
                )
            else:
                self.encoder_img_feat_input_proj = torch.nn.Identity()
        # Transformer encoder positional embeddings.
        n_1d_tokens = 0  # for the latent
        if self.use_robot_state:
            n_1d_tokens += config.num_frames
        if self.use_env_state:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        # TODO
        # self.timestep_pos_embed = nn.Embedding(1, config.dim_model)
        if self.use_images:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)
            self.encoder_temp_pos_embed = nn.Parameter(torch.zeros(1, config.num_frames, config.dim_model), requires_grad=False)
            temp_embed = get_1d_sincos_temp_embed(self.encoder_temp_pos_embed.shape[-1], self.encoder_temp_pos_embed.shape[-2])
            self.encoder_temp_pos_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))
            # TODO
            self.context_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)
            self.cam_feat_pos_emb = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)
            
            self.per_camera_pos_emb = nn.Parameter(torch.randn(3, config.dim_model)) # TODO: change 3 to number of cameras
            nn.init.trunc_normal_(self.per_camera_pos_emb, std=0.2)
        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        if (not self.config.use_flow) or self.config.use_flow_as_auxiliary or self.config.use_image_as_auxiliary:
            self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        # else:
            # self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(config.dim_model, config.output_shapes["action"][0])
        if self.config.use_diffusion:
            self.t_embedder = TimestepEmbedder(
                config.dim_model,
            )
            self.noisy_action_embedder = nn.Linear(
                    config.output_shapes["action"][0], config.dim_model
                )
        if self.config.use_flow_matching:
            self.noisy_action_embedder = nn.Linear(
                    config.output_shapes["action"][0], config.dim_model
                )
            self.action_time_embedder = nn.Sequential(
                nn.Linear(config.dim_model*2, config.dim_model),
                nn.SiLU(),
                nn.Linear(config.dim_model, config.dim_model),
            )
        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        if self.config.use_flow_as_auxiliary or self.config.use_image_as_auxiliary:
            modules = [self.encoder.parameters(), self.action_decoder.parameters()]
            if self.config.use_flow_as_auxiliary:
                modules.append(self.flow_decoder.parameters())
            if self.config.use_image_as_auxiliary:
                modules.append(self.image_decoder.parameters())
            for p in chain(*modules):
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        else:
            for p in chain(self.encoder.parameters(), self.decoder.parameters()):
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        `batch` should have the following structure:
        {
            "observation.state" (optional): (B, state_dim) batch of robot states.

            "observation.images": (B, n_cameras, C, H, W) batch of images.
                AND/OR
            "observation.environment_state": (B, env_dim) batch of environment states.

            "action" (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        """
        if self.config.use_vae and self.training:
            assert (
                "action" in batch
            ), "actions must be provided when using the variational objective in training mode."
        # print(batch["observation.images"])
        batch_size = (
            batch["observation.images"]
            if "observation.images" in batch
            else batch["observation.environment_state"]
        ).shape[0]

        # Prepare the latent for input to the transformer encoder.
        if self.config.use_vae and "action" in batch:
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            if self.use_robot_state:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch["observation.state"])
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(batch["action"])  # (B, S, D)

            if self.use_robot_state:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # Prepare fixed positional embedding.
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # Prepare key padding mask for the transformer encoder. We have 1 or 2 extra tokens at the start of the
            # sequence depending whether we use the input states or not (cls and robot state)
            # False means not a padding token.
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.use_robot_state else 1),
                False,
                device=batch["observation.state"].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )  # (bs, seq+1 or 2)

            # Forward pass through VAE encoder to get the latent PDF parameters.
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # select the class token, with shape (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            # Sample the latent with the reparameterization trick.
            # latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
            # latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
            #     batch["observation.environment_state"].device
            # )

        # Prepare transformer encoder inputs.
        # encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        encoder_in_tokens = []
        # encoder_in_pos_embed = []
        # Robot state token.
        if self.use_robot_state:
            for i in range(batch["observation.state"].shape[1]):
                encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch["observation.state"][:,i]))
        # Environment state token.
        if self.use_env_state:
            encoder_in_tokens.append(
                self.encoder_env_state_input_proj(batch["observation.environment_state"])
            )

        # Camera observation features and positional embeddings.
        if self.use_images:
            all_cam_features = []
            all_cam_pos_embeds = []
            if self.config.token_as_input:
                encoder_in_tokens.extend(einops.rearrange(batch["observation.images"], "b n c-> n b c"))
                encoder_in_pos_embed.extend(einops.rearrange(torch.zeros_like(batch["observation.images"][0]).unsqueeze(0), "b n c -> n b c"))
            else:
                ####################### For ResNet or DinoV2 ########################
                if self.config.vision_backbone == "resnet18" or self.config.vision_backbone == "dino_v2" or self.config.vision_backbone == "dino_v3" or self.config.vision_backbone == "dino_v3_lora":
                    for cam_index in range(batch["observation.images"].shape[-4]):
                        if self.config.use_film:
                            cam_features = self.backbone(batch["observation.images"][:, cam_index], task_emb=batch["observation.context_emb"])["feature_map"]
                        else:
                            cam_features = self.backbone(batch["observation.images"][:, cam_index])["feature_map"]
                        # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use
                        # buffer
                        cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                        cam_pos_embed = cam_pos_embed + self.per_camera_pos_emb[cam_index].unsqueeze(-1).unsqueeze(-1)  # Add per-camera positional embedding.
                        cam_features = self.encoder_img_feat_input_proj(cam_features)  # (B, C, h, w)
                        # temp_embed = self.encoder_temp_pos_embed[:, cam_index].view(1, self.config.dim_model, 1, 1)
                        # # Add the temporal positional embedding to the camera feature map.
                        # cam_features = cam_features + temp_embed
                        all_cam_features.append(cam_features)
                        all_cam_pos_embeds.append(cam_pos_embed)
                ######################## For VGGt #########################
                elif self.config.vision_backbone == "vggt":
                    cam_features = self.backbone(batch["observation.images"])["feature_map"]
                    for cam_index in range(cam_features.shape[1]):
                        cam_feat = cam_features[:, cam_index]
                        cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_feat).to(dtype=cam_feat.dtype)
                        # cam_features_proj = self.encoder_img_feat_input_proj(cam_feat)  # (B, C, h, w)
                        cam_pos_embed = torch.zeros_like(cam_pos_embed)
                        all_cam_features.append(cam_feat)
                        all_cam_pos_embeds.append(cam_pos_embed)
                    # Concatenate camera observation feature maps and positional embeddings along the width dimension,
                    # and move to (sequence, batch, dim).
                all_cam_features = torch.cat(all_cam_features, axis=-1)
                encoder_in_tokens.extend(einops.rearrange(all_cam_features, "b c h w -> (h w) b c"))
                all_cam_pos_embeds = torch.cat(all_cam_pos_embeds, axis=-1)
                encoder_in_pos_embed.extend(einops.rearrange(all_cam_pos_embeds, "b c h w -> (h w) b c"))
        
        if "observation.context" in batch:
            if self.config.use_cross_attention:
                # Use the context as a memory for cross-attention.
                memory_in_tokens = []
                memory_in_pos_embed = []
                memory_in_tokens.extend(einops.rearrange(batch["observation.context"], "b t c h w -> (t h w) b c"))
                # print(torch.zeros_like(batch["observation.context"][0]).unsqueeze(0).repeat(batch_size, 1, 1, 1).shape)
                memory_in_pos_embed.extend(torch.zeros((1, 1, batch_size, self.config.dim_model), device=batch["observation.context"].device))
                # memory_in_pos_embed.extend(einops.rearrange(self.context_pos_embed(batch["observation.context"][:, 0]), "b c h w -> (h w) b c"))
                memory_in_tokens = torch.stack(memory_in_tokens, axis=0)
                memory_in_pos_embed = torch.stack(memory_in_pos_embed, axis=0)
            else:
                encoder_in_tokens.extend(einops.rearrange(batch["observation.context"], "b t c h w -> (t h w) b c"))
                encoder_in_pos_embed.extend(einops.rearrange(torch.zeros_like(batch["observation.context"][0]).unsqueeze(0), "b t c h w -> (t h w) b c"))
            if self.config.use_detection_as_auxiliary:
                detection_decoder_in = torch.zeros(
                (1, batch_size, self.config.dim_model),
                dtype=memory_in_pos_embed.dtype,
                device=memory_in_pos_embed.device,
            )
                detection_decoder_out = self.detection_decoder(
                    detection_decoder_in,
                    memory_in_tokens,
                    decoder_pos_embed=self.detection_decoder_pos_embed.weight.unsqueeze(1),
                    encoder_pos_embed=None,
                )
                detection = self.detection_head(detection_decoder_out)
                detection = detection.transpose(0, 1)  # Move back to (B, S, C).

        # Stack all tokens along the sequence dimension.
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)
        ################
        if len(encoder_in_tokens.shape) == 4:
            encoder_in_tokens = encoder_in_tokens.squeeze(2)
        #######################
        # Forward pass through the transformer modules.
        if self.config.use_cross_attention and not self.config.use_spatial_temporal_encoder:
            encoder_out, attn_weights = self.encoder(
                encoder_in_tokens,
                memory=memory_in_tokens,
                query_pos_embed=encoder_in_pos_embed,
                memory_pos_embed=None,
            )
        elif self.config.use_cross_attention and self.config.use_spatial_temporal_encoder:
            encoder_in_tokens = einops.rearrange(encoder_in_tokens, '(t h w) b c -> b t c h w', h=self.config.image_size[0] // self.config.patch_size, w=self.config.image_size[1] // self.config.patch_size)
            memory_in_tokens = einops.rearrange(memory_in_tokens, 'n b c -> b n c')
            encoder_out = self.encoder(
                        encoder_in_tokens,
                        memory_in_tokens,
                        )
            encoder_out = einops.rearrange(encoder_out, 'b f t c -> (f t) b c')
        else:
            encoder_out = self.encoder(encoder_in_tokens, 
                                       pos_embed=encoder_in_pos_embed)
        action_timestep_embed = None
        image_timestep_embed = None
        if "action_timesteps" in batch or "image_timesteps" in batch or "timesteps" in batch:
            action_timestep_source = batch.get("action_timesteps", batch.get("timesteps"))
            image_timestep_source = batch.get("image_timesteps", batch.get("timesteps"))
            if action_timestep_source is not None:
                action_timestep_embed = create_sinusoidal_pos_embedding_pi0(
                    action_timestep_source, self.config.dim_model, min_period=4e-3, max_period=4.0, device=encoder_in_pos_embed.device
                )
            if image_timestep_source is not None:
                image_timestep_embed = create_sinusoidal_pos_embedding_pi0(
                    image_timestep_source, self.config.dim_model, min_period=4e-3, max_period=4.0, device=encoder_in_pos_embed.device
                )

        # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
        if self.config.use_flow_as_auxiliary or self.config.use_image_as_auxiliary:
            action_decoder_in = torch.zeros(
                (self.config.chunk_size, batch_size, self.config.dim_model),
                dtype=encoder_in_pos_embed.dtype,
                device=encoder_in_pos_embed.device,
            )
            if self.config.use_diffusion and "noisy_action" in batch:
                noisy_action = batch["noisy_action"].permute(1, 0, 2)
                action_decoder_in = self.noisy_action_embedder(noisy_action)
            if self.config.use_flow_matching and "noisy_action" in batch:
                noisy_action = batch["noisy_action"]
                action_embed = self.noisy_action_embedder(noisy_action)
                expanded_timestep_embed = action_timestep_embed[:, None, :].expand_as(action_embed).to(dtype=action_embed.dtype)
                action_decoder_in = self.action_time_embedder(torch.cat([action_embed, expanded_timestep_embed], dim=-1)).permute(1, 0, 2)
            action_decoder_out = self.action_decoder(
                action_decoder_in,
                encoder_out,
                encoder_pos_embed=encoder_in_pos_embed,
                decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
            )
            # Move back to (B, S, C).
            action_decoder_out = action_decoder_out.transpose(0, 1)

            actions = self.action_head(action_decoder_out)

            outputs = []
            if self.config.use_flow_as_auxiliary:
                flow_decoder_in = torch.zeros(
                    (batch_size, self.config.chunk_size, 2, self.config.image_size[0], self.config.image_size[1]),
                    dtype=encoder_in_pos_embed.dtype,
                    device=encoder_in_pos_embed.device,
                )
                if self.config.use_diffusion and "noisy_flow" in batch:
                    flow_decoder_in = batch["noisy_flow"]
                if self.config.use_flow_matching and "noisy_flow" in batch:
                    flow_decoder_in = batch["noisy_flow"]
                if (self.config.use_flow_matching or self.config.use_diffusion) and self.config.use_adaLN:
                    flow_dtype = flow_decoder_in.dtype
                    flow_timestep_embed = action_timestep_embed.to(dtype=flow_dtype)
                    flow_decoder_out = self.flow_decoder(
                        encoder_out,
                        flow_decoder_in,
                        flow_timestep_embed,
                    )
                else:
                    flow_decoder_out = self.flow_decoder(
                        encoder_out,
                        flow_decoder_in,
                    )
                outputs.append(flow_decoder_out)

            if self.config.use_image_as_auxiliary:
                image_decoder_in = torch.zeros(
                    (
                        batch_size,
                        self.config.image_aux_num_views,
                        self.config.image_aux_channels,
                        self.config.image_size[0],
                        self.config.image_size[1],
                    ),
                    dtype=encoder_in_pos_embed.dtype,
                    device=encoder_in_pos_embed.device,
                )
                if "noisy_image" in batch:
                    image_decoder_in = batch["noisy_image"]
                image_decoder_out = self.image_decoder(
                    encoder_out,
                    image_decoder_in,
                    timestep_embed=image_timestep_embed,
                )
                outputs.append(image_decoder_out)

            if len(outputs) == 1:
                return actions, outputs[0], (mu, log_sigma_x2)
            return actions, tuple(outputs), (mu, log_sigma_x2)
        
        elif not self.config.use_flow:
            decoder_in = torch.zeros(
                (self.config.chunk_size, batch_size, self.config.dim_model),
                dtype=encoder_in_pos_embed.dtype,
                device=encoder_in_pos_embed.device,
            )
            if self.config.use_diffusion and "noisy_action" in batch:
                noisy_action = batch["noisy_action"]
                decoder_in = self.noisy_action_embedder(noisy_action)
            if self.config.use_flow_matching and "noisy_action" in batch:
                noisy_action = batch["noisy_action"]
                action_embed = self.noisy_action_embedder(noisy_action)
                action_timestep_embed = action_timestep_embed[:, None, :].expand_as(action_embed).to(dtype=action_embed.dtype)
                decoder_in = self.action_time_embedder(torch.cat([action_embed, action_timestep_embed], dim=-1)).permute(1, 0, 2)
            decoder_out = self.decoder(
                decoder_in,
                encoder_out,
                encoder_pos_embed=encoder_in_pos_embed,
                decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
            )
            # Move back to (B, S, C).
            decoder_out = decoder_out.transpose(0, 1)

            actions = self.action_head(decoder_out)
            if self.config.use_detection_as_auxiliary:
                return actions, detection, (mu, log_sigma_x2)
            elif self.config.supervise_attn_weights:
                return actions, attn_weights, (mu, log_sigma_x2)
            else:
                return actions, (mu, log_sigma_x2)
        
        else:
            decoder_in = torch.zeros(
                (batch_size, self.config.chunk_size, 2, self.config.image_size[0], self.config.image_size[1]),
                dtype=encoder_in_pos_embed.dtype,
                device=encoder_in_pos_embed.device,
            ) 
            decoder_out = self.decoder(
                encoder_out,
                decoder_in,
                # self.decoder_pos_embed.weight.unsqueeze(1),
            )

            return decoder_out, (mu, log_sigma_x2)

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
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask_2d, dropout_p=self.attn_drop.p if self.training else 0.0).reshape(B, Nq, Cq)
            else:
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0).reshape(B, Nq, Cq) # require pytorch 2.0

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

class ACTEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # note: [0] to select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x

class ACTAdaptiveFlowDecoder(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.decoder = AdaptiveSpatialCrossAttentionDecoder(
            d_model=config.dim_model,
            nhead=config.n_heads,
            num_layers=config.n_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            patch_size=config.patch_size,
            batch_first=True,
            num_frames=config.chunk_size,  
            in_channels=config.flow_channel,
        )
        self.out_channels = config.flow_channel
        self.chunk_size = config.chunk_size
        self.patch_size = config.patch_size
        self.final_layer = UnconditionalFinalLayer(config.dim_model, config.patch_size, 
                                                   config.flow_channel)
    
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, encoder_out, decoder_in, c):
        """
        encoder_out: (B, S, C)
        decoder_in: (B, T, C)
        decoder_pos_embed: (1, T, C)
        """
        batches = decoder_in.shape[0]
        encoder_out = einops.rearrange(encoder_out, 'f b d -> b f d')
        x = self.decoder(decoder_in, encoder_out, c)
        x = einops.rearrange(x, 'b f t d -> (b f) t d', b=batches)
        x = self.final_layer(x)
        x = self.unpatchify(x)
        x = einops.rearrange(x, '(b f) c h w -> b f c h w', b=batches)
        return x
    

class ACTFlowDecoder(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.decoder = SpatialCrossAttentionDecoder(
            d_model=config.dim_model,
            nhead=config.n_heads,
            num_layers=config.n_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            patch_size=config.patch_size,
            batch_first=True,
            num_frames=config.chunk_size,  
            in_channels=config.flow_channel,
        )
        self.out_channels = config.flow_channel
        self.chunk_size = config.chunk_size
        self.patch_size = config.patch_size
        self.final_layer = UnconditionalFinalLayer(config.dim_model, config.patch_size, 
                                                   config.flow_channel)
    
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, encoder_out, decoder_in):
        """
        encoder_out: (B, S, C)
        decoder_in: (B, T, C)
        decoder_pos_embed: (1, T, C)
        """
        batches = decoder_in.shape[0]
        encoder_out = einops.rearrange(encoder_out, 'f b d -> b f d')
        x = self.decoder(decoder_in, encoder_out)
        x = einops.rearrange(x, 'b f t d -> (b f) t d', b=batches)
        x = self.final_layer(x)
        x = self.unpatchify(x)
        x = einops.rearrange(x, '(b f) c h w -> b f c h w', b=batches)
        return x


class ACTImageDecoder(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config
        self.num_views = config.image_aux_num_views
        self.out_channels = config.image_aux_channels
        self.patch_size = config.patch_size
        self.x_embedder = PatchEmbed(config.image_size, config.patch_size, self.out_channels, config.dim_model)
        num_patches = self.x_embedder.num_patches
        self.num_patches = num_patches

        self.layers = nn.ModuleList([ACTImageDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)
        self.final_layer = FinalLayer(config.dim_model, config.patch_size, self.out_channels)

        self.patch_pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.dim_model), requires_grad=False)
        patch_pos_embed = get_2d_sincos_pos_embed(
            self.patch_pos_embed.shape[-1], int(num_patches ** 0.5)
        )
        self.patch_pos_embed.data.copy_(torch.from_numpy(patch_pos_embed).float().unsqueeze(0))
        self.view_pos_embed = nn.Parameter(torch.randn(1, self.num_views, 1, config.dim_model))
        nn.init.trunc_normal_(self.view_pos_embed, std=0.02)

    def unpatchify(self, x: Tensor) -> Tensor:
        c = self.out_channels
        p = self.patch_size
        h = self.config.image_size[0] // p
        w = self.config.image_size[1] // p
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, w * p))

    def forward(self, encoder_out: Tensor, decoder_in: Tensor, timestep_embed: Tensor | None = None) -> Tensor:
        batches, num_views = decoder_in.shape[:2]
        if num_views != self.num_views:
            raise ValueError(
                f"Expected {self.num_views} auxiliary image views, got {num_views}."
            )
        x = einops.rearrange(decoder_in, "b v c h w -> (b v) c h w")
        x = self.x_embedder(x)
        x = einops.rearrange(x, "(b v) p d -> b v p d", b=batches, v=num_views)
        x = x + self.patch_pos_embed.unsqueeze(1) + self.view_pos_embed[:, :num_views]
        if timestep_embed is None:
            timestep_embed = torch.zeros(
                batches,
                self.config.dim_model,
                dtype=x.dtype,
                device=x.device,
            )
        else:
            timestep_embed = timestep_embed.to(dtype=x.dtype, device=x.device)

        memory = einops.rearrange(encoder_out, "s b d -> b s d")
        for layer in self.layers:
            x = layer(x, memory, timestep_embed)

        x = self.norm(x)
        x = einops.rearrange(x, "b v p d -> (b v) p d")
        conditioning = timestep_embed.repeat_interleave(num_views, dim=0)
        x = self.final_layer(x, conditioning)
        x = self.unpatchify(x)
        x = einops.rearrange(x, "(b v) c h w -> b v c h w", b=batches, v=num_views)
        return x


class ACTImageDecoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        if config.attention_mode == "math":
            self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout, batch_first=True)
            self.cross_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout, batch_first=True)
        elif config.attention_mode == "flash":
            self.self_attn = Attention(
                dim=config.dim_model,
                num_heads=config.n_heads,
                attn_drop=config.dropout,
                attention_mode="flash",
            )
            self.cross_attn = Attention(
                dim=config.dim_model,
                num_heads=config.n_heads,
                attn_drop=config.dropout,
                attention_mode="flash",
            )
        else:
            raise ValueError(f"Unsupported attention mode {config.attention_mode}")

        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.dim_model, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(config.dim_model, elementwise_affine=False, eps=1e-6)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.dim_model, 9 * config.dim_model, bias=True),
        )

    def forward(self, x: Tensor, memory: Tensor, c: Tensor) -> Tensor:
        bsz, num_views, num_patches, dim = x.shape
        shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(9, dim=1)
        )

        skip = x
        x_sa = modulate(self.norm1(x), shift_sa.unsqueeze(1), scale_sa.unsqueeze(1))
        x_view = einops.rearrange(x_sa, "b v p d -> (b v) p d")
        if isinstance(self.self_attn, nn.MultiheadAttention):
            x_view = self.self_attn(x_view, x_view, x_view)[0]
        else:
            x_view = self.self_attn(x_view, x_view, x_view, batch_first=True)
        x_view = einops.rearrange(x_view, "(b v) p d -> b v p d", b=bsz, v=num_views)
        x = skip + gate_sa.view(bsz, 1, 1, dim) * self.dropout1(x_view)

        skip = x
        x_ca = modulate(self.norm2(x), shift_ca.unsqueeze(1), scale_ca.unsqueeze(1))
        x_seq = einops.rearrange(x_ca, "b v p d -> b (v p) d")
        if isinstance(self.cross_attn, nn.MultiheadAttention):
            x_seq = self.cross_attn(x_seq, memory, memory)[0]
        else:
            x_seq = self.cross_attn(x_seq, memory, memory, batch_first=True)
        x_seq = einops.rearrange(x_seq, "b (v p) d -> b v p d", v=num_views, p=num_patches)
        x = skip + gate_ca.view(bsz, 1, 1, dim) * self.dropout2(x_seq)

        skip = x
        x_mlp = modulate(self.norm3(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        x_mlp = self.linear2(self.dropout(self.activation(self.linear1(x_mlp))))
        x = skip + gate_mlp.view(bsz, 1, 1, dim) * self.dropout3(x_mlp)
        return x
    
class ACTDecoder(nn.Module):
    def __init__(self, config: ACTConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        if config.attention_mode == "math":
            self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
            self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        elif config.attention_mode == "flash":
            self.self_attn = Attention(
                dim=config.dim_model,   
                num_heads=config.n_heads,
                attn_drop=config.dropout,
                attention_mode='flash',
            )
            self.multihead_attn = Attention(
                dim=config.dim_model,   
                num_heads=config.n_heads,
                attn_drop=config.dropout,
                attention_mode='flash',
            )
        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            decoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            encoder_pos_embed: (DS, 1, C) Positional_embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


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

class ACTCrossAttentionEncoder(nn.Module):
    """Convenience module for running multiple cross-attention encoder layers followed by normalization."""

    def __init__(self, config: ACTConfig):
        super().__init__()
        num_layers = config.n_encoder_layers
        if config.supervise_attn_weights:
            self.layers = nn.ModuleList([ACTCrossAttentionEncoderLayer(config) if i < num_layers - 1 else LastACTCrossAttentionEncoderLayer(config) for i in range(num_layers)])
        else:
            self.layers = nn.ModuleList([ACTCrossAttentionEncoderLayer(config) for _ in range(num_layers)])
        self.ret_attn_weights = config.supervise_attn_weights
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        query_pos_embed: Tensor | None = None,
        memory_pos_embed: Tensor | None = None,
    ) -> Tensor:
        if not self.ret_attn_weights:
            for layer in self.layers:
                x = layer(x, memory, tgt_key_padding_mask, memory_key_padding_mask, query_pos_embed, memory_pos_embed)
            if self.norm is not None:
                x = self.norm(x)
            return x, None
        else:
            for layer in self.layers[:-1]:
                x = layer(x, memory, tgt_key_padding_mask, memory_key_padding_mask, query_pos_embed, memory_pos_embed)
            x, attn_weights = self.layers[-1](x, memory, tgt_key_padding_mask, memory_key_padding_mask, query_pos_embed, memory_pos_embed)
            if self.norm is not None:
                x = self.norm(x)
            return x, attn_weights
    
class ACTCrossAttentionEncoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        if config.attention_mode == 'math':
            # Self-attention: decoder attending to itself
            self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
            
            # Cross-attention: decoder attending to encoder output
            self.cross_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        elif config.attention_mode == 'flash':
            self.self_attn = Attention(
                dim=config.dim_model,
                num_heads=config.n_heads,
                attn_drop=config.dropout,
                attention_mode='flash'
            )
            self.cross_attn = Attention(
                dim=config.dim_model,
                num_heads=config.n_heads,
                attn_drop=config.dropout,
                attention_mode='flash'
            )

        # Feedforward layers
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        # Normalization layers
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)

        # Dropouts
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(
        self,
        x: Tensor,                                # Decoder input (with positional embedding already added)
        memory: Tensor,                           # Encoder output
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        query_pos_embed: Tensor | None = None,          # Optional positional embeddings for decoder query
        memory_pos_embed: Tensor | None = None          # Optional positional embeddings for encoder memory
    ) -> Tensor:
        # === Self-Attention ===
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if query_pos_embed is None else x + query_pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=tgt_key_padding_mask)[0]
        x = skip + self.dropout1(x)
        if not self.pre_norm:
            x = self.norm1(x)

        # === Cross-Attention ===
        skip = x
        if self.pre_norm:
            x = self.norm2(x)
        q = x
        k = memory if memory_pos_embed is None else memory + memory_pos_embed
        cross_attn_output = self.cross_attn(q, k, value=memory, key_padding_mask=memory_key_padding_mask)[0]
        x = skip + self.dropout2(cross_attn_output)
        if not self.pre_norm:
            x = self.norm2(x)

        # === Feedforward ===
        skip = x
        if self.pre_norm:
            x = self.norm3(x)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)

        return x

class LastACTCrossAttentionEncoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        if config.attention_mode == 'math':
            # Self-attention: decoder attending to itself
            self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
            
            # Cross-attention: decoder attending to encoder output
            self.cross_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        elif config.attention_mode == 'flash':
            self.self_attn = Attention(
                dim=config.dim_model,
                num_heads=config.n_heads,
                attn_drop=config.dropout,
                attention_mode='flash'
            )
            self.cross_attn = Attention(
                dim=config.dim_model,
                num_heads=config.n_heads,
                attn_drop=config.dropout,
                attention_mode='math'
            )

        # Feedforward layers
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        # Normalization layers
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)

        # Dropouts
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(
        self,
        x: Tensor,                                # Decoder input (with positional embedding already added)
        memory: Tensor,                           # Encoder output
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        query_pos_embed: Tensor | None = None,          # Optional positional embeddings for decoder query
        memory_pos_embed: Tensor | None = None          # Optional positional embeddings for encoder memory
    ) -> Tensor:
        # === Self-Attention ===
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if query_pos_embed is None else x + query_pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=tgt_key_padding_mask)[0]
        x = skip + self.dropout1(x)
        if not self.pre_norm:
            x = self.norm1(x)

        # === Cross-Attention ===
        skip = x
        if self.pre_norm:
            x = self.norm2(x)
        q = x
        k = memory if memory_pos_embed is None else memory + memory_pos_embed
        cross_attn_output, attn_weights = self.cross_attn(q, k, value=memory, key_padding_mask=memory_key_padding_mask, return_averaged_attn=True)
        x = skip + self.dropout2(cross_attn_output)
        if not self.pre_norm:
            x = self.norm2(x)

        # === Feedforward ===
        skip = x
        if self.pre_norm:
            x = self.norm3(x)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)

        return x, attn_weights
