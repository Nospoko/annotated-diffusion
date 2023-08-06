from dataclasses import dataclass

@dataclass
class Hyperparameters:
    dataset: str # huggingface dataset
    batch_size: int
    num_workers: int
    lr: float # weight decay
    l2: float # loss type: mse, mae, huber
    loss_type: str
    num_epochs: int
    accelerator: str
    precision: int
    overfit_batches: int
    save_every_n_steps: int # how often the model is saved

@dataclass
class Paths:
    save_ckpt_dir: str # directory where checkpoints will be saved
    load_ckpt_path: str # if not None, specifies path to model state dict which will be loaded
    log_dir: str

@dataclass
class Unet:
    dim: int # initial number of unet channels (will be multiplied by dim_mults)
    dim_mults: tuple[int] # tuple of dim multipliers, it also specifies network depth
    in_channels: int # image channels
    num_resnet_groups: int

@dataclass
class ForwardDiffusion:
    beta_start: float
    beta_end: float
    timesteps: int
    schedule_type: str # schedule type: cosine, linear, quadratic, sigmoid

@dataclass
class Models:
    unet: Unet
    forward_diffusion: ForwardDiffusion

@dataclass
class Logger:
    run_name: str
    log_every_n_steps: int

@dataclass
class Config:
    hyperparameters: Hyperparameters
    paths: Paths
    models: Models
    logger: Logger