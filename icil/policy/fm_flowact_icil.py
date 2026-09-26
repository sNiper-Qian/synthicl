import torch
from torch import nn, Tensor
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from typing import Tuple, Optional
from icil.config.policy_config import PolicyConfig
from icil.config.shared_config import SharedConfig
from icil.config.configuration_act import ACTConfig
from icil.common.utils import get_activation_fn, create_sinusoidal_pos_embedding, PositionalEncoding, sample_points_on_polygon, render_action_on_image
from icil.common.vision import ResNetBackbone, DINOv2BackBone, DinoVGGTFusionBackbone, DINOv3BackBone
from icil.common.dino import DINOv3FeatureExtractor
from timm.layers import Mlp
from icil.policy.latte import Latte, UnconditionalFinalLayer
from icil.policy.act import ACT, ACTSinusoidalPositionEmbedding2d
import einops
import numpy as np
from icil.common.resnet_film import ResNetFilmBackbone
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
import torchvision.transforms as T
from torchvision.transforms import v2
import time

class CUDAMemoryTracer:
    def __init__(self, device=0):
        self.device = device
        self.last_alloc = torch.cuda.memory_allocated(self.device)
        self.start_time = time.time()
        self.events = []

    def log(self, tag):
        torch.cuda.synchronize(self.device)
        cur = torch.cuda.memory_allocated(self.device)
        res = torch.cuda.memory_reserved(self.device)
        peak = torch.cuda.max_memory_allocated(self.device)
        delta = cur - self.last_alloc
        t = time.time() - self.start_time
        self.events.append((t, tag, cur, res, peak, delta))
        self.last_alloc = cur
        print(f"[{t:7.2f}s] {tag:<18} "
              f"alloc={cur/1e6:7.1f}MB  reserved={res/1e6:7.1f}MB  "
              f"peak={peak/1e6:7.1f}MB  Δ={delta/1e6:+6.1f}MB")
        
class FmFlowActICIL(nn.Module):
    def __init__(self, config: PolicyConfig, shared_config: SharedConfig, act_config: ACTConfig):
        '''
        Initializes the ICIL model.
        Parameters:
        config : PolicyConfig : The configuration object for the ICIL model.
        '''
        super().__init__()
        # self.memory_tracer = CUDAMemoryTracer()
        # TODO
        self.config = config
        self.n_modalities = 1 # len(shared_config.image_keys)
        self.sampling_interval = shared_config.sampling_interval
        if shared_config.prompt_length % self.sampling_interval != 0:
            num_input_token_encoder = (shared_config.prompt_length // self.sampling_interval + 1) * (self.n_modalities) # + shared_config.num_traj_per_task # TODO
        else:
            num_input_token_encoder = (shared_config.prompt_length // self.sampling_interval) * (self.n_modalities) # + shared_config.num_traj_per_task
        self.use_film = act_config.use_film
        
        print(f"Using {config.vision_backbone} backbone (with film: {self.use_film})")

        if self.use_film:
            # Mapping from the output of the encoder to the context embedding
            self.context_embedder = nn.Sequential(
                nn.Conv2d(config.dim_model, act_config.film_embedding_dim, kernel_size=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )

        self.eos_token = nn.Parameter(torch.randn(1, 3, config.image_size[0], config.image_size[1]))
        self.cls_token = nn.Parameter(torch.randn(1, 3, config.image_size[0], config.image_size[1]))
        self.pooling_strategy = config.pooling_strategy

        self.pos_encoder = PositionalEncoding(config.dim_model)

        # if config.decoder_type == "cross_attention":
        #     self.action_predictor = SpatialCrossAttentionActionPredictor(config, shared_config)
        # else:
        #     raise ValueError(f"Decoder type {config.decoder_type} not supported.")

        self.device = shared_config.device
        self.n_pred_steps = shared_config.n_pred_steps
        self.n_hist_steps = shared_config.n_hist_steps
        self.obs_dim = config.obs_dim
        self.img_size = config.image_size
        self.action_channels = config.action_channels
        self.decoder_type = config.decoder_type
        self.single_step_observation = shared_config.single_step_observation
        self.has_obs = len(shared_config.obs_keys) > 0
        self.has_image = len(shared_config.image_keys) > 0
        self.has_bg = shared_config.bg_key != ""
        self.img_keys = shared_config.image_keys
        self.vision_backbone = config.vision_backbone
        self.patch_size = config.patch_size
        self.task_length = shared_config.task_length
        self.prompt_length = shared_config.prompt_length
        self.num_train_timesteps = config.num_train_timesteps
        self.num_inference_timesteps = config.num_inference_timesteps
        self.n_samples_per_task = shared_config.n_samples_per_task
        self.use_flow_as_auxiliary = act_config.use_flow_as_auxiliary
        self.use_image_as_auxiliary = act_config.use_image_as_auxiliary
        self.use_image_flow_matching = act_config.use_image_flow_matching
        self.use_detection_as_auxiliary = act_config.use_detection_as_auxiliary
        self.image_aux_loss_weight = act_config.image_aux_loss_weight
        self.supervise_attn_weights = act_config.supervise_attn_weights
        self.attn_loss_weight = act_config.attn_loss_weight
        self.tokens_per_frame = len(shared_config.image_keys) * self.img_size[0] // config.patch_size * self.img_size[1] // config.patch_size # + 2 for action and observation tokens
        self.context_norm = nn.LayerNorm(config.dim_model, elementwise_affine=False, eps=1e-6)
        self.cam_feat_pos_emb = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)
        self.per_camera_pos_emb = nn.Parameter(torch.randn(len(shared_config.image_keys), 1, config.dim_model))
        nn.init.trunc_normal_(self.per_camera_pos_emb, std=0.2)
        if self.has_obs:
            self.obs_pos_emb = nn.Parameter(torch.randn(config.dim_model))
            nn.init.trunc_normal_(self.obs_pos_emb, std=0.2)
        self.action_pos_emb = nn.Parameter(torch.randn(config.dim_model))
        nn.init.trunc_normal_(self.action_pos_emb, std=0.2)

        self.noise_scheduler = self._make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        self.encoder = Latte(
            input_size=(config.image_size[0], config.image_size[1]),
            patch_size=config.patch_size,
            # in_channels=config.image_channels,
            in_channels=config.dim_model,  
            depth=config.n_encoder_layers,
            num_heads=config.n_heads,
            hidden_size=config.dim_model,
            num_frames=num_input_token_encoder,
            # use_resnet=config.vision_backbone == "resnet18",
            token_input=True,  # We are using tokens as input
            tokens_per_frame=self.tokens_per_frame,
        )

        self.normalize_img = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std= [0.229, 0.224, 0.225]
        )
        self.resize_img = v2.Resize(config.image_size, antialias=True)
        # self.normalize_img = T.Normalize(
        #     mean=[0, 0, 0],
        #     std= [1, 1, 1]
        # )
        self.transform = self._make_transforms()
        if config.vision_backbone == "resnet18":
            backbone_model = getattr(torchvision.models, "resnet18")(
                        replace_stride_with_dilation=[False, False, False],
                        weights="ResNet18_Weights.IMAGENET1K_V1",
                        norm_layer=FrozenBatchNorm2d,
                    )
            for param in backbone_model.parameters():
                param.requires_grad = False
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
            self.encoder_img_feat_input_proj = nn.Conv2d(
                        512, config.dim_model, kernel_size=1
                    )
        elif config.vision_backbone == "dino_v2":
            self.backbone = DINOv2BackBone(config)
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.encoder_img_feat_input_proj = nn.Conv2d(
                    self.backbone.num_channels, config.dim_model, kernel_size=1
                )
        elif config.vision_backbone == "dino_v3":
            self.backbone = DINOv3BackBone(config)
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.encoder_img_feat_input_proj = nn.Conv2d(
                    self.backbone.num_channels, config.dim_model, kernel_size=1
                )
        elif config.vision_backbone == "vggt":
            self.backbone = DinoVGGTFusionBackbone(config)
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif config.vision_backbone == "dino_v3_lora":
            self.backbone = DINOv3FeatureExtractor(model_name='dinov3_vits16_plus', 
                                                    device="cuda", 
                                                    use_multi_layer=False, 
                                                    lora_rank=32,
                                                    use_lora=True,
                                                    lora_alpha=32.0,
                                                    lora_dropout=0.0,
                                                    lora_target_modules=["q", "v"],      # adapt query/value projections
                                                    lora_layers=[8, 9, 10, 11],          # last 4 transformer blocks
                                                    )
            self.encoder_img_feat_input_proj = nn.Conv2d(
                    self.backbone.feature_dim, config.dim_model, kernel_size=1
                )
        self.act = ACT(act_config, self.backbone)
        if self.has_obs:
            self.obs_encoder = nn.Linear(config.obs_dim, config.dim_model)
        self.action_encoder = nn.Linear(config.action_dim, config.dim_model)

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_beta(self, alpha, beta, bsize, device):
        gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
        gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
        return gamma1 / (gamma1 + gamma2)

    def _sample_time(self, bsize, device, mode, alpha, beta, time_min, time_max):
        if mode == "beta":
            raw_time = self.sample_beta(alpha, beta, bsize, device)
        elif mode == "uniform":
            raw_time = torch.rand((bsize,), device=device)
        elif mode == "high_noise_beta":
            raw_time = self.sample_beta(beta, alpha, bsize, device)
        else:
            raise ValueError(f"Unsupported time_sampling_mode {mode}")
        time = raw_time * (time_max - time_min) + time_min
        return time.to(dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        return self.sample_action_time(bsize, device)

    def sample_action_time(self, bsize, device):
        return self._sample_time(
            bsize,
            device,
            self.config.action_time_sampling_mode,
            self.config.action_time_sampling_alpha,
            self.config.action_time_sampling_beta,
            self.config.action_time_min,
            self.config.action_time_max,
        )

    def sample_image_time(self, bsize, device):
        return self._sample_time(
            bsize,
            device,
            self.config.image_time_sampling_mode,
            self.config.image_time_sampling_alpha,
            self.config.image_time_sampling_beta,
            self.config.image_time_min,
            self.config.image_time_max,
        )

    def merge_prompt(
        self, 
        observation : torch.Tensor, 
        images : list[torch.Tensor],
        action : torch.Tensor,
        inference : bool = False,
    ) -> torch.Tensor:
        # Downsample images
        # print(observation.shape, images[0].shape, action.shape)
        observation = observation[:, ::self.sampling_interval] if self.has_obs else None
        action = action[:, ::self.sampling_interval]

        if self.has_image:
            frames = []
            for i, imgs in enumerate(images):
                imgs = imgs[:, ::self.sampling_interval]  # Downsample images by sampling interval
                imgs = imgs.permute(0, 1, 4, 2, 3)
                imgs = self._preprocess_images(imgs, augment=not inference)
                B, T, C, H, W = imgs.shape
                imgs = imgs[:, :, None, :, :, :] # B, T, 1, C, H, W
                # if i == 1:
                #     # dropout for the second camera view to encourage the model to use multiple views
                #     dropout_mask = torch.rand(B, T, 1, 1, 1, 1, device=imgs.device) > 0.5
                #     imgs = imgs * dropout_mask
                frames.append(imgs)
            frames = torch.cat(frames, dim=2) # B, T, len(image_keys), C, H, W
            # print(observation.shape, frames.shape, action.shape)
            B, T, P, C, H, W = frames.shape  # P perspectives
            
            ####### For ResNet backbone ########
            if self.vision_backbone == "resnet18" or self.vision_backbone == "dino_v2" or self.vision_backbone == "dino_v3" or self.vision_backbone == "dino_v3_lora":
                # Merge batch, time, perspective dims for CNN
                x = einops.rearrange(frames, 'b t p c h w -> (b t p) c h w')  # (B*T*P, C, H, W)
                feat_map = self.backbone(x)["feature_map"]                # (B*T*P, C_feat, H_feat, W_feat)
                feat_map = self.encoder_img_feat_input_proj(feat_map)     # (B*T*P, D, H_feat, W_feat)
                # mask for the second camera view to encourage the model to use multiple views during training
                # if not inference and P > 1:
                #     feat_map = feat_map.view(B, T, P, feat_map.shape[1], feat_map.shape[2], feat_map.shape[3])  # (B, T, P, D, H_feat, W_feat)
                #     dropout_mask = torch.rand(B, T, 1, 1, feat_map.shape[4], feat_map.shape[5], device=feat_map.device) > 0.5
                #     feat_map = feat_map * dropout_mask
                #     feat_map = feat_map.view(B*T*P, feat_map.shape[3], feat_map.shape[4], feat_map.shape[5])  # (B*T*P, D, H_feat, W_feat)
                # Add positional encoding to the per-camera feature map
                cam_feat_pos_embed = self.cam_feat_pos_emb(feat_map).to(feat_map.device)  # (1, D, H_feat, W_feat)
                feat_map = feat_map + cam_feat_pos_embed  # (B*T*P, D, H_feat, W_feat)
                # # Reshape to (B*T, P, D, H_feat, W_feat)
                # feat_map = einops.rearrange(feat_map, '(b t p) c h w -> (b t) p c h w', b=B, t=T, p=P)  # (B*T, P, D, H_feat, W_feat)
                
            ####### For VGGT backbone ########
            elif self.vision_backbone == "vggt":
                x = einops.rearrange(frames, 'b t p c h w -> (b t p) c h w')  # (B*T, P, C, H, W)
                feat_map = self.backbone(x)["feature_map"]                # (B*T, P, C_feat, H_feat, W_feat)
                # feat_map = einops.rearrange(feat_map, '(b t) p c h w -> (b t p) c h w', b=B, t=T, p=P) # (B*T*P, C_feat, H_feat, W_feat)
                # feat_map = self.encoder_img_feat_input_proj(feat_map)     # (B*T*P, D, H_feat, W_feat)
            
            BP, D, Hf, Wf = feat_map.shape
            N = Hf * Wf
            # Flatten spatial dims and project
            tokens = feat_map.permute(0, 2, 3, 1).reshape(BP, N, D)  # (B*T*P, N, D)
            # Reshape back to (B, T, P, N, D) and merge perspectives into tokens axis
            img_tokens = einops.rearrange(tokens, '(b t p) n d -> b t p n d', b=B, t=T, p=P, n=N, d=D)  # (B, T, P, N, D)    
            
            # Add positional encoding for each perspective  
            if self.vision_backbone == "resnet18" or self.vision_backbone == "dino_v2" or self.vision_backbone == "dino_v3":
                img_tokens = img_tokens + self.per_camera_pos_emb  # (B, T, P, N, D) # TODO: Uncomment if using Resnet   
            img_tokens = img_tokens.reshape(B, T, P * N, D)  # (B, T, P*N, D)

        if self.has_obs:
            obs_embed = self.obs_encoder(observation).unsqueeze(2)
            obs_embed = obs_embed + self.obs_pos_emb

        # Concatenate all tokens: [img_tokens, action, obs]
        if self.has_image:
            out = img_tokens
        else:
            out = obs_embed
        # out shape: (B, T, P*N + 2, D)
        return out

    def preprocess_prompt(
        self, 
        observation : torch.Tensor, 
        images : list[torch.Tensor],
        action : torch.Tensor,
        eos_idx : list[int],
        padding_mask : torch.Tensor,
        insert_cls : bool = True,
        sampling_interval: int = 1,
    ) -> torch.Tensor:
        """
        Preprocess the visual context demonstration for SynthICL.
        
        Parameters:
        observation : B, T, obs_dim (batch size, timesteps, obs_dim)
        action : B, T, action_dim (batch size, timesteps, action_dim)
        eos_idx : list[int] : The list of indices where the prompt ends.
        padding_mask : B, T : The padding mask for the prompt data.
        
        Returns:
        torch.Tensor: The preprocessed data, a tensor of shape (B, T * 2, self.latent_dim).
        """
        # # observation processing
        # if self.has_obs:
        #     f_s = self.obs_encoder(observation) # B, T, self.latent_dim
        #     f_s = f_s[:, :, None, :] # B, T, 1, self.latent_dim
        
        # image processing
        if self.has_image:
            f_is = []
            for i, imgs in enumerate(images):
                # crop to 112*112
                imgs = imgs[:, :, :, :, :]
                imgs = imgs.permute(0, 1, 4, 2, 3)
                B, T, C, H, W = imgs.shape
                imgs = imgs[:, :, None, :, :, :] # B, T, 1, C, H, W
                f_is.append(imgs)
            f_is = torch.cat(f_is, dim=2) # B, T, len(image_keys), C, H, W
        if sampling_interval > 1:
            # ensure last frames are included
            f_is = f_is[:, ::sampling_interval]
        f_sia = f_is
        # # Stack all images
        # f_sia = einops.rearrange(f_is, 'b t n c h w -> b (t n) c h w') # B, T, n_modalities, C*H*W
        
        # # aggregate all modalities
        # if self.has_obs and self.has_image:
        #     f_sia = torch.cat([f_s, f_is, f_a], dim=2)
        # elif self.has_obs:
        #     f_sia = torch.cat([f_s, f_a], dim=2)
        # elif self.has_image:
        #     # TODO 
        #     f_is = f_is.squeeze(2)
        #     f_a = f_a.squeeze(2)
        #     f_sia = torch.cat([f_is, f_a], dim=2)
        # else:
        #     raise ValueError("No observation or image data provided.")
        # f_sia = f_is.squeeze(2) # B, T, n_modalities + 1, C, H, W
        # f_sia: B, T, n_modalities + 1, C, H, W
        # f_sia = f_sia.view(f_sia.shape[0], -1, f_sia.shape[-3], f_sia.shape[-2], f_sia.shape[-1]) # B, T * (n_modalities + 1), C, H, W                     
        # n_sia = self.n_modalities + 1
        n_sia = 1

        # add eos token and padding mask
        eos_embed = self.eos_token.repeat(1, f_sia.shape[2], 1, 1, 1)
        new_f_sia = []
        new_padding_mask = []
        for i in range(f_sia.shape[0]):  
            data = f_sia[i]
            data_pd_mask = torch.repeat_interleave(padding_mask[i], repeats=n_sia) # repeat the mask for state and action
            data_eos_idx = eos_idx[i] // sampling_interval + 1 
            for idx in reversed(data_eos_idx):
                idx = int(idx)
                if idx != 1: 
                    # Cuz the indices are reversed, adding eos token doesn't change the indices
                    data = torch.cat([data[:idx * n_sia], eos_embed, data[idx * n_sia:]], dim=0)
                    data_pd_mask = torch.cat([data_pd_mask[:idx * n_sia], torch.full((1,), False, dtype=torch.bool, device=self.device), data_pd_mask[idx * n_sia:]], dim=0)
                else:
                    # Pad with eos token at the end for uniforming the length
                    data = torch.cat([data, eos_embed], dim=0)
                    data_pd_mask = torch.cat([data_pd_mask, torch.full((1,), True, dtype=torch.bool, device=self.device)], dim=0)
            new_f_sia.append(data)
            new_padding_mask.append(data_pd_mask)
        f_sia = torch.stack(new_f_sia)
        new_padding_mask = torch.stack(new_padding_mask)
        if insert_cls:
            # add cls token
            cls_embed = einops.repeat(
                    self.cls_token, "1 d -> b 1 d", b=f_sia.shape[0]
                )
            f_sia = torch.cat([cls_embed, f_sia], dim=1)
            new_padding_mask = torch.cat([torch.full((f_sia.shape[0], 1), False, dtype=torch.bool, device=self.device), new_padding_mask], dim=1)
        return f_sia, new_padding_mask

    def _make_transforms(self,) -> T.Compose:
        transforms = []
        transforms.append(v2.ColorJitter(brightness=[0.8, 1.2], 
                                        contrast=[0.8, 1.2], 
                                        saturation=[0.8, 1.2], 
                                        hue=[-0.02, 0.02]),
                                        )
        # transforms.append(v2.RandomErasing(scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random', inplace=False))
        # transforms.append(v2.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 2.0)))
        return T.Compose(transforms)

    def _preprocess_images(self, images: torch.Tensor, augment: bool) -> torch.Tensor:
        """Resize and augment RGB values before applying ImageNet normalization."""
        images = self.resize_img(images)
        if augment:
            images = self.transform(images)
        return self.normalize_img(images)

    def _make_noise_scheduler(self, name: str, **kwargs: dict) -> DDPMScheduler | DDIMScheduler:
        """
        Factory for noise scheduler instances of the requested type. All kwargs are passed
        to the scheduler.
        """
        if name == "DDPM":
            return DDPMScheduler(**kwargs)
        elif name == "DDIM":
            return DDIMScheduler(**kwargs)
        else:
            raise ValueError(f"Unsupported noise scheduler type {name}")
    
    def inference(self, data : dict, return_aux_images: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        This function runs the ICIL model in inference
        Parameters:
        data : dict : The input data dictionary.
        Returns:
        torch.Tensor: Predicted action
        """
        prompt_observation = data["prompt_observation_seq"].float() if self.has_obs else None
        prompt_images = []
        if self.has_image:
            for key in self.img_keys:
                prompt_images.append(data[f"prompt_{key}"])
        prompt_action = data["prompt_action_seq"].float()
        observation = data["observation"].float() if self.has_obs else None
        images = []
        if self.has_image:
            for key in self.img_keys:
                image = data[key].permute(0, 1, 2, 5, 3, 4)
                images.append(self._preprocess_images(image, augment=False))
        action = data["action"]
        f_sa = self.merge_prompt(prompt_observation,
                                 prompt_images, 
                                 prompt_action, 
                                 inference=True,
                                )
        # f_sa, new_padding_mask = self.preprocess_prompt(prompt_observation,
        #                                                 prompt_images, 
        #                                                 prompt_action, 
        #                                                 prompt_eos_idx, 
        #                                                 padding_mask, 
        #                                                 insert_cls=self.pooling_strategy == "cls",
        #                                                 sampling_interval=self.sampling_interval)
        # feat_embs = []
        # for i in range(len(images)):
        #     feat_emb = self.encoder(f_sa[:, :, i])
        #     feat_embs.append(feat_emb)
        # feat_emb = torch.concatenate(feat_embs, dim=1)
        feat_emb = self.encoder(f_sa)
        feat_emb = self.context_norm(feat_emb)
        # feat_emb = einops.rearrange(feat_emb, 'b t (h w) c -> b t h w c', h = self.img_size[0] // self.patch_size, w = self.img_size[1] // self.patch_size)
        feat_emb = einops.rearrange(feat_emb, 'b t (h w) c -> b t h w c', h=1, w=self.tokens_per_frame)
        feat_emb = einops.rearrange(feat_emb, 'b t h w c -> b t c h w')
        # # mean pooling
        # feat_emb = feat_emb.mean(dim=1).unsqueeze(1)
        if self.use_film:
            context_emb = self.context_embedder(feat_emb.squeeze(1))
        if self.has_image:
            images = torch.stack(images, dim=2)[:, :1]  # B, T, N, C, H, W
            images = einops.rearrange(images, 'b t n p c h w -> b t (n p) c h w')  # B, T, N * P, C, H, W
            images = einops.rearrange(images, 'b n t c h w -> (b n) t c h w')
        if self.has_obs:
            observation = einops.rearrange(observation[:, :1], 'b n np c -> (b n) np c')
        batch = {
            # "observation.images": images,
            "observation.context": feat_emb,
        }
        if self.has_image:
            batch["observation.images"] = images
        if self.use_film:
            batch["observation.context_emb"] = context_emb
        if self.has_obs:
            batch["observation.state"] = observation
        # Sample prior.
        action_shape = (action.shape[0], self.n_pred_steps, action.shape[-1])
        noise = self.sample_noise(
            shape=action_shape,
            device=self.device,
        )
        image_x_t = None
        if return_aux_images:
            if not self.use_image_as_auxiliary:
                raise ValueError("Image auxiliary decoder is not enabled.")
            if self.use_image_flow_matching:
                image_x_t = self.sample_noise(
                    shape=(
                        action.shape[0],
                        self.act.config.image_aux_num_views,
                        self.act.config.image_aux_channels,
                        self.img_size[0],
                        self.img_size[1],
                    ),
                    device=self.device,
                )
        dt = -1.0 / self.num_inference_timesteps
        dt = torch.tensor(dt, dtype=torch.float32, device=self.device)

        x_t = noise
        for step in range(self.num_inference_timesteps):
            time = 1.0 + step * dt
            expanded_time = time.expand(action_shape[0])
            batch = {
                # "observation.images": images,
                "observation.context": feat_emb,
                "action_timesteps": expanded_time,
                "noisy_action": x_t,
            }
            if return_aux_images and self.use_image_flow_matching:
                batch["image_timesteps"] = expanded_time
                batch["noisy_image"] = image_x_t
            if self.has_image:
                batch["observation.images"] = images
            if self.use_film:
                batch["observation.context_emb"] = context_emb
            if self.has_obs:
                batch["observation.state"] = observation
            if (
                not self.use_flow_as_auxiliary
                and not self.use_image_as_auxiliary
                and not self.use_detection_as_auxiliary
                and not self.supervise_attn_weights
            ):
                v_t, _ = self.act(batch)
                pred_image_velocity = None
            else:
                v_t, aux_output, _ = self.act(batch)
                if return_aux_images:
                    pred_image_velocity = aux_output[-1] if isinstance(aux_output, tuple) else aux_output
                else:
                    pred_image_velocity = None

            # Euler step
            x_t += dt * v_t
            if return_aux_images and self.use_image_flow_matching:
                image_x_t += dt * pred_image_velocity

        if return_aux_images:
            if self.use_image_flow_matching:
                return x_t, image_x_t

            image_batch = {
                "observation.context": feat_emb,
                "action_timesteps": torch.zeros(action_shape[0], dtype=torch.float32, device=self.device),
                "noisy_action": x_t,
            }
            if self.has_image:
                image_batch["observation.images"] = images
            if self.use_film:
                image_batch["observation.context_emb"] = context_emb
            if self.has_obs:
                image_batch["observation.state"] = observation
            _, pred_images, _ = self.act(image_batch)
            return x_t, pred_images
        return x_t
        
    def select_context_images(self, images: torch.Tensor, ctx_idx: torch.Tensor) -> torch.Tensor:
        """
        images:  (B, N, K, 1, 3, H, W)
        ctx_idx: (B, N)
        
        Returns:
            context_images: (B, N, K, 1, 3, H, W)
            For each position n, selects the image at the last timestep
            belonging to the same context group.
        """
        B, N = ctx_idx.shape
        device = ctx_idx.device

        # Find context-group boundaries along N
        change = torch.ones_like(ctx_idx, dtype=torch.bool)
        change[:, 1:] = ctx_idx[:, 1:] != ctx_idx[:, :-1]

        # group_id: 0,0,1,1,1,2,2,...
        group_id = change.cumsum(dim=1) - 1   # (B, N)

        # Number of groups per batch
        num_groups = group_id[:, -1] + 1
        max_groups = num_groups.max().item()

        # Count how many elements in each group
        counts = torch.zeros(B, max_groups, dtype=torch.long, device=device)
        counts.scatter_add_(
            dim=1,
            index=group_id,
            src=torch.ones_like(group_id, dtype=torch.long)
        )

        # Last timestep position of each group in the original sequence
        last_pos = counts.cumsum(dim=1) - 1   # (B, max_groups)

        # For each timestep, map to its group's last position
        selected_pos = last_pos.gather(dim=1, index=group_id)   # (B, N)

        # Gather images along dimension 1 (the sequence dimension N)
        # Expand selected_pos to match all trailing dims
        index = selected_pos.view(B, N, 1, 1, 1, 1, 1).expand_as(images)
        context_images = torch.gather(images, dim=1, index=index)

        return context_images

    def predict_aux_images_for_logging(self, data: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_image_as_auxiliary:
            raise ValueError("Image auxiliary decoder is not enabled.")
        if not self.has_image:
            raise ValueError("Image auxiliary logging requires image observations.")

        prompt_observation = data["prompt_observation_seq"].float() if self.has_obs else None
        prompt_images = []
        if self.has_image:
            for key in self.img_keys:
                prompt_images.append(data[f"prompt_{key}"])
        prompt_action = data["prompt_action_seq"].float()
        observation = data["observation"].float() if self.has_obs else None
        images = []
        if self.has_image:
            for key in self.img_keys:
                image = data[key].permute(0, 1, 2, 5, 3, 4)
                images.append(self._preprocess_images(image, augment=False))

        f_sa = self.merge_prompt(
            prompt_observation,
            prompt_images,
            prompt_action,
            inference=True,
        )
        feat_emb = self.encoder(f_sa)
        feat_emb = self.context_norm(feat_emb)
        feat_emb = einops.rearrange(feat_emb, 'b t (h w) c -> b t h w c', h=1, w=self.tokens_per_frame)
        feat_emb = einops.rearrange(feat_emb, 'b t h w c -> b t c h w')
        if self.use_film:
            context_emb = self.context_embedder(feat_emb.squeeze(1))

        task_padding_mask = data["task_padding_mask"].float()
        if len(task_padding_mask.shape) == 2:
            task_lengths = (~task_padding_mask.bool()).sum(dim=1)
        else:
            valid_steps = (task_padding_mask[:, :, 0] == 1).any(dim=-1).any(dim=-1)
            task_lengths = valid_steps.sum(dim=1)

        images = torch.stack(images, dim=2)  # B, T, N, C, H, W
        assigned_wps = data["auxiliary"].long()
        image_targets = self.select_context_images(images, assigned_wps)
        image_targets = einops.rearrange(image_targets, 'b n k p c h w -> b n (k p) c h w')
        images = einops.rearrange(images, 'b t n p c h w -> b t (n p) c h w')

        step_indices = []
        for ub in task_lengths:
            ub = max(int(ub), 1)
            step_indices.append(torch.randint(0, ub, (1,), device=images.device))
        step_indices = torch.stack(step_indices, dim=0)
        batch_idx = torch.arange(images.shape[0], device=images.device).unsqueeze(1)

        images = images[batch_idx, step_indices]
        images = einops.rearrange(images, 'b n t c h w -> (b n) t c h w')
        image_targets = image_targets[batch_idx, step_indices]
        image_targets = einops.rearrange(image_targets, 'b n p c h w -> (b n) p c h w')

        if self.has_obs:
            observation = observation[batch_idx, step_indices]
            observation = einops.rearrange(observation, 'b n np c -> (b n) np c')

        action = data["action"][batch_idx, step_indices]
        action = einops.rearrange(action, 'b n np c -> (b n) np c')
        if self.use_image_flow_matching:
            action_shape = action.shape
            image_shape = image_targets.shape
            action_x_t = self.sample_noise(shape=action_shape, device=self.device)
            image_x_t = self.sample_noise(shape=image_shape, device=self.device)
            dt = -1.0 / self.num_inference_timesteps
            dt = torch.tensor(dt, dtype=torch.float32, device=self.device)

            for step in range(self.num_inference_timesteps):
                time = 1.0 + step * dt
                expanded_time = time.expand(action_shape[0])
                batch = {
                    "observation.context": feat_emb,
                    "action_timesteps": expanded_time,
                    "image_timesteps": expanded_time,
                    "noisy_action": action_x_t,
                    "observation.images": images,
                    "noisy_image": image_x_t,
                }
                if self.use_film:
                    batch["observation.context_emb"] = context_emb
                if self.has_obs:
                    batch["observation.state"] = observation

                pred_action_velocity, pred_image_velocity, _ = self.act(batch)
                action_x_t = action_x_t + dt * pred_action_velocity
                image_x_t = image_x_t + dt * pred_image_velocity

            return image_x_t, image_targets, images

        action_x_t = self.sample_noise(shape=action.shape, device=self.device)
        dt = -1.0 / self.num_inference_timesteps
        dt = torch.tensor(dt, dtype=torch.float32, device=self.device)

        for step in range(self.num_inference_timesteps):
            time = 1.0 + step * dt
            expanded_time = time.expand(action.shape[0])
            batch = {
                "observation.context": feat_emb,
                "action_timesteps": expanded_time,
                "noisy_action": action_x_t,
                "observation.images": images,
            }
            if self.use_film:
                batch["observation.context_emb"] = context_emb
            if self.has_obs:
                batch["observation.state"] = observation

            pred_action_velocity, _, _ = self.act(batch)
            action_x_t = action_x_t + dt * pred_action_velocity

        batch = {
            "observation.context": feat_emb,
            "action_timesteps": torch.zeros(action.shape[0], dtype=torch.float32, device=self.device),
            "noisy_action": action_x_t,
            "observation.images": images,
        }
        if self.use_film:
            batch["observation.context_emb"] = context_emb
        if self.has_obs:
            batch["observation.state"] = observation
        _, pred_images, _ = self.act(batch)
        return pred_images, image_targets, images
    
    def forward(self, data : dict) -> torch.Tensor:
        """
        This function runs the ICIL model.
        
        Parameters:
        data : dict : The input data dictionary.
        
        Returns:
        torch.Tensor: Predicted action
        torch.Tensor: Loss
        """
        prompt_observation = data["prompt_observation_seq"].float() if self.has_obs else None
        prompt_images = []
        if self.has_image:
            for key in self.img_keys:
                prompt_images.append(data[f"prompt_{key}"])
        prompt_action = data["prompt_action_seq"].float()
        prompt_eos_idx = data["prompt_eos_idx"]
        padding_mask = data["prompt_padding_mask"]
        observation = data["observation"].float() if self.has_obs else None
        images = []
        if self.has_image:
            for key in self.img_keys:
                image = data[key].permute(0, 1, 2, 5, 3, 4)
                images.append(self._preprocess_images(image, augment=True))
        action = data["action"]
        if self.use_flow_as_auxiliary:
            flow = data["auxiliary"]
        if self.supervise_attn_weights:
            assigned_wps = data["auxiliary"].long()
            B, N = assigned_wps.shape
            assigned_wps = einops.rearrange(assigned_wps, 'b n -> (b n)')
            start = (assigned_wps - 1) * self.tokens_per_frame     # (B,)
            end   = (assigned_wps + 1) * self.tokens_per_frame     # (B,)
            attn_weights_mask = torch.ones(B*N,
                                          self.tokens_per_frame,
                                          self.prompt_length * self.tokens_per_frame,
                                          dtype=torch.bool,
                                          device=self.device)
            arange_k = torch.arange(self.prompt_length * self.tokens_per_frame, device=self.device).unsqueeze(0)  # (1, T_k)
            selected_1d = (arange_k >= start[:, None]) & (arange_k < end[:, None])  # (B, T_k)
            selected_3d = selected_1d.unsqueeze(1).expand(-1, self.tokens_per_frame, -1)
            attn_weights_mask[selected_3d] = False   # Broadcast across T_q
            # === Create weights ===
            attn_weights_gt = (~attn_weights_mask).float()   # 1 where selected, 0 otherwise
            # print(attn_weights_gt)    
            # print(attn_weights_mask)
            attn_weights_gt = einops.rearrange(attn_weights_gt, '(b n) t_q t_k -> b n t_q t_k', b=B, n=N)
            attn_weights_mask = einops.rearrange(attn_weights_mask, '(b n) t_q t_k -> b n t_q t_k', b=B, n=N)
        if self.has_bg:
            bg = data["backgrounds"]
        # self.memory_tracer.log("Start preprocess")
        f_sa = self.merge_prompt(prompt_observation,
                                 prompt_images, 
                                 prompt_action, 
                                 inference=False,
                                )
        # self.memory_tracer.log("End merge prompt")
        # feat_embs = []
        # for i in range(len(images)):
        #     feat_emb = self.encoder(f_sa[:, :, i])
        #     feat_embs.append(feat_emb)
        # feat_emb = torch.concatenate(feat_embs, dim=1)
        feat_emb = self.encoder(f_sa)
        # self.memory_tracer.log("End encoder")
        feat_emb = self.context_norm(feat_emb)
        # feat_emb = einops.rearrange(feat_emb, 'b t (h w) c -> b t h w c', h = self.img_size[0] // self.patch_size, w = self.img_size[1] // self.patch_size)
        feat_emb = einops.rearrange(feat_emb, 'b t (h w) c -> b t h w c', h=1, w=self.tokens_per_frame)

        n_samples = self.n_samples_per_task
        # repeat the feature embedding for the number of task length
        feat_emb = einops.repeat(feat_emb, 'b t h w c -> b n t h w c', n=n_samples)
        feat_emb = einops.rearrange(feat_emb, 'b n t h w c -> (b n) t c h w')
        if self.use_film:
            context_emb = self.context_embedder(feat_emb.squeeze(1))
   
        # randomly sample an image from the task_length
        task_padding_mask = data["task_padding_mask"].float()
        if len(task_padding_mask.shape) == 2:
            task_lengths = (~task_padding_mask.bool()).sum(dim=1)
        else:
            valid_steps = (task_padding_mask[:, :, 0] == 1).any(dim=-1).any(dim=-1)
            task_lengths = valid_steps.sum(dim=1)
        # sample step index for each batch from 0 to task_length
        if self.has_image:
            images = torch.stack(images, dim=2)  # B, T, N, C, H, W
            if self.use_image_as_auxiliary:
                assigned_wps = data["auxiliary"].long()
                B, N = assigned_wps.shape
                # assigned_wps = einops.rearrange(assigned_wps, 'b n -> (b n)')
                # image_targets = torch.stack(images, dim=2)
                image_targets = self.select_context_images(images, assigned_wps)  # (B, N, K, 1, 3, H, W)
                image_targets = einops.rearrange(image_targets, 'b n k p c h w -> b n (k p) c h w')  # B, N, K*P, C, H, W
            images = einops.rearrange(images, 'b t n p c h w -> b t (n p) c h w')  # B, T, N * P, C, H, W
            
        # Sample training samples
        step_indices = []
        for ub in task_lengths:
            ub = max(int(ub), 1)
            if ub >= n_samples:
                samples = torch.randperm(ub, device=action.device)[:n_samples]
            else:
                samples = torch.randint(0, ub, (n_samples,), device=action.device)
            step_indices.append(samples)
        step_indices = torch.stack(step_indices, dim=0)
        # step_indices = [random.sample(range(ub), n_samples) for ub in task_lengths]
        if self.has_image:
            batch_idx = torch.arange(images.shape[0], device=images.device).unsqueeze(1).expand(-1, n_samples)
        else:
            batch_idx = torch.arange(observation.shape[0], device=observation.device).unsqueeze(1).expand(-1, n_samples)
        action = action[batch_idx, step_indices]
        if self.use_detection_as_auxiliary:
            detection_label = detection_label[batch_idx, step_indices]
            detection_label = einops.rearrange(detection_label, 'b n c -> (b n) c')
        if self.use_flow_as_auxiliary:
            flow = flow[batch_idx, step_indices]
            flow = einops.rearrange(flow, 'b n np h w c -> (b n) np c h w')
            action = einops.rearrange(action, 'b n np c -> (b n) np c')
        else:
            action = einops.rearrange(action, 'b n np c -> (b n) np c')
        if self.has_image:
            images = images[batch_idx, step_indices]
            images = einops.rearrange(images, 'b n t c h w -> (b n) t c h w')
        if self.has_obs:
            observation = observation[batch_idx, step_indices]
            observation = einops.rearrange(observation, 'b n np c -> (b n) np c')
        if self.has_bg:
            bg = bg[batch_idx, step_indices]
            bg = einops.rearrange(bg, 'b n c h w -> (b n) c h w')
        if self.supervise_attn_weights:
            attn_weights_mask = attn_weights_mask[batch_idx, step_indices]
            attn_weights_mask = einops.rearrange(attn_weights_mask, 'b n t_q t_k -> (b n) t_q t_k')
            attn_weights_gt = attn_weights_gt[batch_idx, step_indices]
            attn_weights_gt = einops.rearrange(attn_weights_gt, 'b n t_q t_k -> (b n) t_q t_k')
        if self.use_flow_as_auxiliary:
            masks = task_padding_mask[batch_idx, step_indices]
            masks = einops.rearrange(masks, 'b n np h w -> (b n) np h w')
            masks = einops.repeat(masks, 'b np h w -> b np t h w', t=self.action_channels)
        if self.use_image_as_auxiliary:
            image_targets = image_targets[batch_idx, step_indices]
            image_targets = einops.rearrange(image_targets, 'b n p c h w -> (b n) p c h w')
        # Sample flow matching steps
        action_time = self.sample_action_time(
            action.shape[0],
            device=self.device,
        )

        # Sample noise to add to the trajectory.
        noise = self.sample_noise(
            shape=action.shape,
            device=self.device,
        )

        time_expanded = action_time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * action
        x_t = x_t.to(dtype=time_expanded.dtype, device=self.device)
        u_t = noise - action

        if self.use_image_as_auxiliary and self.use_image_flow_matching:
            if image_targets is None:
                raise ValueError("Image auxiliary training requires image observations.")
            image_time = self.sample_image_time(
                image_targets.shape[0],
                device=self.device,
            )
            image_time_expanded = image_time[:, None, None, None, None]
            image_noise = self.sample_noise(
                shape=image_targets.shape,
                device=self.device,
            )
            noisy_image = image_time_expanded * image_noise + (1 - image_time_expanded) * image_targets
            noisy_image = noisy_image.to(dtype=image_targets.dtype, device=self.device)
            image_velocity_target = image_noise - image_targets

        batch = {
            # "observation.images": images,
            "observation.context": feat_emb,
            "action_timesteps": action_time,
            "noisy_action": x_t,
        }
        if self.has_image:
            batch["observation.images"] = images
        if self.use_image_as_auxiliary:
            if self.use_image_flow_matching:
                batch["image_timesteps"] = image_time
                batch["noisy_image"] = noisy_image
        # if self.use_flow_as_auxiliary:
        #     batch["noisy_flow"] = f_t
        if self.use_film:
            batch["observation.context_emb"] = context_emb
        if self.has_obs:
            batch["observation.state"] = observation
        if self.use_flow_as_auxiliary:
            v_t, pred_flow, _ = self.act(batch)
            flow_loss = F.l1_loss(pred_flow, flow, reduction='none')
            flow_loss = flow_loss * masks
            flow_loss = flow_loss.sum() / masks.sum()
            action_loss = F.mse_loss(v_t, u_t)
            loss = action_loss + flow_loss
            return v_t, loss, (action_loss, flow_loss)
        elif self.use_image_as_auxiliary:
            v_t, pred_image, _ = self.act(batch)
            if self.use_image_flow_matching:
                image_loss = F.mse_loss(pred_image, image_velocity_target)
            else:
                image_loss = F.mse_loss(pred_image, image_targets)
            action_loss = F.mse_loss(v_t, u_t)
            loss = action_loss + self.image_aux_loss_weight * image_loss
            return v_t, loss, (action_loss, self.image_aux_loss_weight * image_loss)
        elif self.use_detection_as_auxiliary:
            v_t, pred_detection, _ = self.act(batch)
            detection_label = detection_label.unsqueeze(1)
            detection_loss = F.l1_loss(pred_detection, detection_label)
            action_loss = F.mse_loss(v_t, u_t)
            loss = action_loss + detection_loss
            return v_t, loss, (action_loss, detection_loss)
        elif self.supervise_attn_weights:
            v_t, attn_weights, _ = self.act(batch)
            attn_loss = F.mse_loss(attn_weights, attn_weights_gt, reduction='none')
            attn_loss = attn_loss.masked_fill(~attn_weights_mask, 0.0)
            attn_loss = attn_loss.sum() / (attn_weights_mask).sum()
            action_loss = F.mse_loss(v_t, u_t)
            loss = action_loss + attn_loss * self.attn_loss_weight
            return v_t, loss, (action_loss, attn_loss*self.attn_loss_weight)
        else:          
            v_t, _ = self.act(batch)
            # self.memory_tracer.log("End ACT forward")
            loss = F.mse_loss(v_t, u_t)
            return v_t, loss, (loss, torch.tensor(0.0, device=self.device))
