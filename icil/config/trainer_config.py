from dataclasses import dataclass
@dataclass
class TrainerConfig:
    epochs : int = 500
    pin_memory : bool = True
    num_workers : int = 20 
    eval_freq : int = 1000
    save_freq : int = 1000
    ckpt_dir : str = "./ckpt"   
    lr: float = 1e-4
    refresh_freq: int = 10000
    grad_clip_norm: float = 1.0  