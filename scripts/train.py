"""Train the SynthICL policy on generated pseudo-demonstrations."""

from __future__ import annotations

import argparse
import datetime
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SynthICL on HDF5 episodes from the pseudo-demo generator."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing episode_*.h5 files (subdirectories are scanned recursively).",
    )
    parser.add_argument(
        "--dino-repo",
        type=Path,
        required=True,
        help="Local clone of facebookresearch/dinov3.",
    )
    parser.add_argument(
        "--dino-weights",
        type=Path,
        required=True,
        help="DINOv3 ViT-S/16+ pretrained checkpoint.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--pretrained-path", type=Path)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of tasks per loader batch; each task contributes 8 sampled steps.",
    )
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--eval-frequency", type=int, default=1000)
    parser.add_argument("--save-frequency", type=int, default=5000)
    parser.add_argument("--train-split", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Train without logging to Weights & Biases.",
    )
    parser.add_argument("--wandb-project", default="synthicl")
    parser.add_argument("--wandb-name")
    args = parser.parse_args()

    if not args.dataset_dir.is_dir():
        parser.error(f"--dataset-dir is not a directory: {args.dataset_dir}")
    if not any(args.dataset_dir.rglob("*.h5")):
        parser.error(f"--dataset-dir contains no HDF5 episodes: {args.dataset_dir}")
    if not args.dino_repo.is_dir():
        parser.error(f"--dino-repo is not a directory: {args.dino_repo}")
    if not args.dino_weights.is_file():
        parser.error(f"--dino-weights is not a file: {args.dino_weights}")
    if args.pretrained_path is not None and not args.pretrained_path.is_file():
        parser.error(f"--pretrained-path is not a file: {args.pretrained_path}")
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs and batch size must be positive; num workers cannot be negative")
    if args.eval_frequency < 1 or args.save_frequency < 1:
        parser.error("evaluation and save frequencies must be positive")
    if not 0.0 < args.train_split < 1.0:
        parser.error("--train-split must be between 0 and 1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    return args


def make_configs(args: argparse.Namespace):
    from icil.config.configuration_act import ACTConfig
    from icil.config.dataset_config import DatasetConfig
    from icil.config.policy_config import PolicyConfig
    from icil.config.shared_config import SharedConfig
    from icil.config.trainer_config import TrainerConfig

    dataset_config = DatasetConfig(
        hdf5_path=str(args.dataset_dir.resolve()),
        num_prompt_traj=1,
        waypoint_key="waypoints_idx",
        max_n_prompts=1,
        label_keys=[],
        train_val_split=args.train_split,
    )
    model_config = PolicyConfig(
        n_encoder_layers=8,
        dim_model=512,
        dim_feedforward=2048,
        n_heads=8,
        dropout=0.1,
        pre_norm=False,
        pooling_strategy="none",
        patch_size=16,
        decoder_type="cross_attention",
        vision_backbone="dino_v3",
        dino_repo=str(args.dino_repo.resolve()),
        dino_weights=str(args.dino_weights.resolve()),
        image_size=(256, 256),
        action_channels=4,
        obs_dim=7,
        action_dim=7,
        action_time_sampling_mode="beta",
        action_time_sampling_alpha=1.5,
        action_time_sampling_beta=1.0,
        action_time_min=0.001,
        action_time_max=0.999,
        num_inference_timesteps=10,
    )
    shared_config = SharedConfig(
        batch_size=args.batch_size,
        prompt_length=6,
        n_hist_steps=1,
        single_step_observation=False,
        image_size=(256, 256),
        image_keys=["images_front", "images_wrist", "images_side"],
        num_traj_per_task=2,
        action_key="actions",
        task_length=80,
        n_pred_steps=8,
        sampling_interval=1,
        n_samples_per_task=8,
        # SynthICL conditions on RGB only; robot_states remain in HDF5 for statistics/debugging.
        obs_keys=[],
        auxiliary_key="assigned_wps",
        seed=args.seed,
        device=args.device,
    )
    act_config = ACTConfig(
        use_film=False,
        use_cross_attention=True,
        n_encoder_layers=7,
        n_decoder_layers=2,
        use_flow=False,
        patch_size=16,
        dim_model=512,
        dim_feedforward=2048,
        chunk_size=8,
        num_frames=1,
        pre_norm=True,
        use_spatial_temporal_encoder=False,
        use_flow_as_auxiliary=False,
        use_image_as_auxiliary=True,
        use_image_flow_matching=False,
        use_detection_as_auxiliary=False,
        image_aux_num_views=3,
        image_aux_loss_weight=1.0,
        supervise_attn_weights=False,
        input_shapes={"observation.images.top": [3, 256, 256]},
        output_shapes={"action": [7]},
        use_diffusion=False,
        use_flow_matching=True,
        use_adaLN=False,
        vision_backbone="dino_v3",
        image_size=(256, 256),
    )
    trainer_config = TrainerConfig(
        ckpt_dir=str(args.checkpoint_dir.resolve()),
        lr=1e-4,
        epochs=args.epochs,
        num_workers=args.num_workers,
        eval_freq=args.eval_frequency,
        save_freq=args.save_frequency,
        grad_clip_norm=1.0,
    )
    return dataset_config, model_config, shared_config, act_config, trainer_config


def make_image_panel(
    observation: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor
) -> np.ndarray:
    observation_row = torch.cat(list(observation), dim=-1)
    target_row = torch.cat(list(target), dim=-1)
    prediction_row = torch.cat(list(prediction), dim=-1)
    panel = torch.cat([observation_row, target_row, prediction_row], dim=-2)
    return (panel.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)


def log_image_predictions(
    run: Any,
    model: FmFlowActICIL,
    eval_batch: dict,
    global_step: int,
) -> None:
    import wandb

    batch_size = next(v.shape[0] for v in eval_batch.values() if torch.is_tensor(v))
    sample_count = min(20, batch_size)
    indices = torch.randperm(batch_size, device=model.device)[:sample_count]
    sample = {
        key: value[indices] if torch.is_tensor(value) else value
        for key, value in eval_batch.items()
    }
    prediction, target, observation = model.predict_aux_images_for_logging(sample)
    mean = torch.tensor([0.485, 0.456, 0.406], device=model.device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=model.device).view(1, 1, 3, 1, 1)
    prediction = (prediction.float() * std + mean).clamp(0.0, 1.0)
    target = (target.float() * std + mean).clamp(0.0, 1.0)
    observation = (observation.float() * std + mean).clamp(0.0, 1.0)
    images = [
        wandb.Image(
            make_image_panel(observation[i], target[i], prediction[i]),
            caption="observation (top), target subgoal (middle), prediction (bottom)",
        )
        for i in range(sample_count)
    ]
    run.log({"eval/image_predictions": images, "step": global_step})


def train(args: argparse.Namespace) -> None:
    from icil.common.utils import collate_fn, move_to_device
    from icil.data.pseudo_demo_dataset import PseudoDemoDataset
    from icil.policy.fm_flowact_icil import FmFlowActICIL

    dataset_config, model_config, shared_config, act_config, trainer_config = make_configs(args)
    device = torch.device(shared_config.device)
    torch.manual_seed(shared_config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(shared_config.seed)
        torch.cuda.empty_cache()

    run_name = args.wandb_name or datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = Path(trainer_config.ckpt_dir) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    all_configs = {
        "dataset": asdict(dataset_config),
        "model": asdict(model_config),
        "shared": asdict(shared_config),
        "act": asdict(act_config),
        "trainer": asdict(trainer_config),
    }
    (checkpoint_dir / "config.json").write_text(json.dumps(all_configs, indent=2))
    run = None
    if not args.disable_wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, name=run_name, config=all_configs)

    model = FmFlowActICIL(model_config, shared_config, act_config).to(device)
    if args.pretrained_path is not None:
        state_dict = torch.load(args.pretrained_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    train_dataset = PseudoDemoDataset(dataset_config, shared_config, split="train")
    eval_dataset = PseudoDemoDataset(dataset_config, shared_config, split="eval")
    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        raise ValueError(
            "The train/eval split is empty. Generate more episodes or adjust --train-split."
        )
    norm_stats_path = checkpoint_dir / "norm_stats.json"
    train_dataset.save_norm_stats(norm_stats_path)
    # Validation must use the same normalization learned from the training split.
    eval_dataset.load_norm_stats(norm_stats_path)
    loader_options = {
        "batch_size": shared_config.batch_size,
        "num_workers": trainer_config.num_workers,
        "persistent_workers": trainer_config.num_workers > 0,
        "pin_memory": trainer_config.pin_memory,
        "collate_fn": collate_fn,
    }
    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, **loader_options
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, shuffle=False, **loader_options
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=trainer_config.lr)
    amp_enabled = device.type == "cuda"
    scaler = GradScaler(device.type, enabled=amp_enabled)
    global_step = 0
    print(f"Training episodes: {len(train_dataset):,}")
    print(f"Evaluation episodes: {len(eval_dataset):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(trainer_config.epochs):
        for batch in train_loader:
            model.train()
            batch = move_to_device(batch, device, torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=amp_enabled):
                _, loss, detailed_loss = model(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer_config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            if run is not None:
                run.log(
                    {
                        "train/action_loss": detailed_loss[0].item(),
                        "train/subgoal_loss": detailed_loss[1].item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        "step": global_step,
                    }
                )

            if global_step % trainer_config.eval_freq == 0:
                model.eval()
                action_loss = 0.0
                subgoal_loss = 0.0
                image_batch = None
                with torch.no_grad():
                    for eval_batch in eval_loader:
                        eval_batch = move_to_device(eval_batch, device, torch.float32)
                        if image_batch is None:
                            image_batch = eval_batch
                        with autocast(device_type=device.type, enabled=amp_enabled):
                            _, _, detailed_eval_loss = model(eval_batch)
                        action_loss += detailed_eval_loss[0].item()
                        subgoal_loss += detailed_eval_loss[1].item()
                action_loss /= len(eval_loader)
                subgoal_loss /= len(eval_loader)
                print(
                    f"epoch={epoch} step={global_step} "
                    f"eval_action_loss={action_loss:.6f} "
                    f"eval_subgoal_loss={subgoal_loss:.6f}"
                )
                if run is not None:
                    run.log(
                        {
                            "eval/action_loss": action_loss,
                            "eval/subgoal_loss": subgoal_loss,
                            "epoch": epoch,
                            "step": global_step,
                        }
                    )
                    if image_batch is not None:
                        log_image_predictions(run, model, image_batch, global_step)

            if global_step % trainer_config.save_freq == 0:
                torch.save(model.state_dict(), checkpoint_dir / f"model_{global_step}.pth")

    torch.save(model.state_dict(), checkpoint_dir / "model_final.pth")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    train(parse_args())
