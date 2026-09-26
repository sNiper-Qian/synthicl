from dataclasses import dataclass, field
@dataclass
class SharedConfig:
    num_traj_per_task: int = 5
    prompt_length: int = 512
    device: str = "cuda"
    n_pred_steps: int = 4
    n_hist_steps: int = 4
    log_name: str = "icil"
    batch_size: int = 8
    seed: int = 42
    task_length: int = 200
    single_step_observation: bool = False
    image_size: tuple[int, int] = (256, 256)
    image_keys: list[str] = field(default_factory=lambda: [])
    masks_keys: list[str] = field(default_factory=lambda: [])
    obs_keys: list[str] = field(default_factory=lambda: [])
    action_key: str = "actions"
    auxiliary_key: str = ""
    bg_key: str = ""
    sampling_interval: int = 1
    n_samples_per_task: int = 8
