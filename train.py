import os
import uuid

import hydra
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger

from config import Config
from models.utils import Transforms
from models.reverse_diffusion import Unet
from models.forward_diffusion import ForwardDiffusion


class UnetTrainingWrapper(pl.LightningModule):
    def __init__(
        self,
        model: Unet,
        forward_diffusion: ForwardDiffusion,
        lr: float = 1e-3,
        l2: float = 0.01,
        loss_type: str = "mse",
        ckpt_dir: str = "checkpoints",
        save_every_n_steps: int = 100,
    ):
        super().__init__()

        # model
        self.model = model
        self.forward_diffusion = forward_diffusion
        self.timesteps = forward_diffusion.timesteps

        # save hyperparameters (lr, l2, loss_type, ckpt_dir, save_every_n_steps)
        self.save_hyperparameters(ignore=["model", "forward_diffusion"])

        # get loss function
        self.loss_function_module = self._loss_function()

    def _loss_function(self) -> nn.Module:
        if self.hparams["loss_type"] == "mse":
            loss = nn.MSELoss()
        elif self.hparams["loss_type"] == "mae":
            loss = nn.L1Loss()
        elif self.hparams["loss_type"] == "huber":
            loss = nn.HuberLoss()
        else:
            raise NotImplementedError()

        return loss

    def _save_models(self):
        # ckpt specifies directory and name of the file is name of the experiment in wandb
        save_path = f"{self.hparams['ckpt_dir']}/{self.logger.experiment.name}.ckpt"
        # saving models
        torch.save({"model": self.model.state_dict(), "forward_diffusion": self.forward_diffusion.state_dict()}, save_path)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x, t)

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.AdamW(
          params=self.model.parameters(),
          lr=self. hparams["lr"],
          weight_decay=self.hparams["l2"],
        )

        return optimizer

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        x = batch["x"]

        batch_size = x.shape[0]

        # sample t
        t = torch.randint(0, self.timesteps, size=(batch_size,), dtype=torch.long, device=self.device)

        # noise batch
        x_noisy, added_noise = self.forward_diffusion.forward(x, t)

        # get predicted noise
        predicted_noise = self.model.forward(x_noisy, t)

        # get loss value for batch
        loss = self.loss_function_module(predicted_noise, added_noise)

        self.log("loss", loss, on_epoch=True)

        if self.global_step % self.hparams["save_every_n_steps"] == 0:
            self._save_models()

        return loss


def preprocess_dataset(dataset_name: str, batch_size: int, num_workers: int):
    dataset = load_dataset(dataset_name)
    transformed_dataset = dataset.with_transform(transforms).remove_columns("label")

    dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, num_workers=num_workers)

    return dataloader


def transforms(examples):
    transform = Transforms(img_size=28)

    examples["x"] = [transform.img2torch(image.convert("L")) for image in examples["image"]]
    del examples["image"]

    return examples


def makedir_if_not_exists(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


@hydra.main(config_path="configs", config_name="config-fashion-mnist")
def train(cfg: Config):
    # get dataloader
    dataloader = preprocess_dataset(
        filepath=cfg.hyperparameters.dataset,
        batch_size=cfg.hyperparameters.batch_size,
        num_workers=cfg.hyperparameters.num_workers,
    )

    # generate some random id for the run
    run_id = str(uuid.uuid1())[:8]

    # make directory for logs if doesn't exist
    makedir_if_not_exists(cfg.paths.log_dir)

    # initializing logger
    logger = WandbLogger(project="annotated-diffusion", name=f"{cfg.logger.run_name}-{run_id}", save_dir=cfg.paths.log_dir)

    # initializing models
    model = Unet(
        dim=cfg.models.unet.dim,
        dim_mults=cfg.models.unet.dim_mults,
        in_channels=cfg.models.unet.in_channels,
        resnet_block_groups=cfg.models.unet.num_resnet_groups,
    )

    forward_diffusion = ForwardDiffusion(
        beta_start=cfg.models.forward_diffusion.beta_start,
        beta_end=cfg.models.forward_diffusion.beta_end,
        timesteps=cfg.models.forward_diffusion.timesteps,
        schedule_type=cfg.models.forward_diffusion.schedule_type,
    )

    # if checkpoint exists load state dict for model and forward diffusion
    if cfg.paths.load_ckpt_path is not None:
        checkpoint = torch.load(cfg.paths.load_ckpt_path)
        model.load_state_dict(checkpoint["model"])
        forward_diffusion.load_state_dict(checkpoint["forward_diffusion"])

    # make directory for checkpoints if doesn't exist
    makedir_if_not_exists(cfg.paths.save_ckpt_dir)

    # initializng lightning model wrapper that will handle training logic
    model_training_wrapper = UnetTrainingWrapper(
        model,
        forward_diffusion,
        lr=cfg.hyperparameters.lr,
        l2=cfg.hyperparameters.l2,
        loss_type=cfg.hyperparameters.loss_type,
        save_every_n_steps=cfg.hyperparameters.save_every_n_steps,
        ckpt_dir=cfg.paths.save_ckpt_dir,
    )

    # callbacks
    callbacks = [TQDMProgressBar()]

    # initializing trainer with specified hyperparameters
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg.hyperparameters.num_epochs,
        accelerator=cfg.hyperparameters.accelerator,
        precision=cfg.hyperparameters.precision,
        overfit_batches=cfg.hyperparameters.overfit_batches,
        log_every_n_steps=cfg.logger.log_every_n_steps,
    )

    # run training
    trainer.fit(model_training_wrapper, dataloader)


if __name__ == "__main__":
    wandb.login()

    train()

    wandb.finish()
