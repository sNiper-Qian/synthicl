from dataclasses import dataclass
from typing import List
from dataclasses import field
@dataclass
class DatasetConfig:
    hdf5_path: str = "data.hdf5"
    num_tasks: int = 100
    train_val_split: float = 0.95
    num_prompt_traj: int | None = None
    enable_debug: bool = False
    goal_key: str = "goal_pose"
    waypoint_key: str = "waypoints_idx"
    load_to_memory: bool = False
    max_n_prompts: int = 4
    obs_scale: float | List = 15.0
    obs_min: float | List = -1.0
    obs_max: float | List = 1.0
    action_scale: float | List = 2
    action_min: float | List = -1.0
    action_max: float | List = 1.0
    action_std: float | List = 0.1
    auxiliary_scale: float | List = 10.0
    auxiliary_min: float | List = -1.0
    auxiliary_max: float | List = 1.0
    with_respect_to_first_action: bool = False
    label_keys: list[str] = field(default_factory=lambda: ["trajectory_label"])
