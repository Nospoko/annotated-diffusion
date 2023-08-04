import uuid
import argparse

import torch
import wandb
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger

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
        save_every_x_steps: int = 100,
    ):
        super().__init__()

        # model
        self.model = model
        self.forward_diffusion = forward_diffusion
        self.timesteps = forward_diffusion.timesteps

        # save hyperparameters
        self.save_hyperparameters(ignore=["model", "forward_diffusion"])

        # get loss function
        self.loss_function_module = self._loss_function()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x, t)

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

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.AdamW(self.model.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["l2"])

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

        if self.global_step % self.hparams["save_every_x_steps"] == 0:
            self._save_models()

        return loss

    def _save_models(self):
        # ckpt specifies directory and name of the file is name of the experiment in wandb
        save_path = f"{self.hparams['ckpt_dir']}/{self.logger.experiment.name}.ckpt"
        # saving models
        torch.save({"model": self.model.state_dict(), "forward_diffusion": self.forward_diffusion.state_dict()}, save_path)


def preprocess_dataset(filepath: str, batch_size: int):
    dataset = load_dataset(filepath)
    transformed_dataset = dataset.with_transform(transforms).remove_columns("label")

    dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size)

    return dataloader


def transforms(examples):
    transform = Transforms(img_size=28)

    examples["x"] = [transform.img2torch(image.convert("L")) for image in examples["image"]]
    del examples["image"]

    return examples


def train(dataloader: DataLoader, args):
    # generate some random id for the run
    run_id = str(uuid.uuid1())[:8]

    # initializing logger
    logger = WandbLogger(project="annotated-diffusion", name=f"{args.run_name}-{run_id}", save_dir=args.log_dir)

    # initializing models
    model = Unet(dim=args.dim, dim_mults=args.dim_mults, in_channels=args.in_channels, resnet_block_groups=args.num_resnet_groups)

    forward_diffusion = ForwardDiffusion(
        beta_start=args.beta_start, beta_end=args.beta_end, timesteps=args.timesteps, schedule_type=args.schedule_type
    )

    # if checkpoint exists load state dict for model and forward diffusion
    if args.load_ckpt_path is not None:
        checkpoint = torch.load(args.load_ckpt_path)
        model.load_state_dict(checkpoint["model"])
        forward_diffusion.load_state_dict(checkpoint["forward_diffusion"])

    # initializng lightning model wrapper that will handle training logic
    model_training_wrapper = UnetTrainingWrapper(
        model,
        forward_diffusion,
        lr=args.lr,
        l2=args.l2,
        loss_type=args.loss_type,
        ckpt_dir=args.save_ckpt_dir,
        save_every_x_steps=args.save_every_x_steps,
    )

    # callbacks
    callbacks = [TQDMProgressBar()]

    # initializing trainer with specified hyperparameters
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=args.num_epochs,
        accelerator=args.accelerator,
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
        overfit_batches=args.overfit_batches,
    )

    # run training
    trainer.fit(model_training_wrapper, dataloader)


def parse_args():
    parser = argparse.ArgumentParser(prog="annotated-diffusion")
    # dataset
    parser.add_argument("--dataset", type=str, default="fashion_mnist", help="Provide path to HuggingFace dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")

    # load checkpoint
    parser.add_argument("--load_ckpt_path", type=str, default=None, help="Load checkpoint path")

    # unet parameters
    parser.add_argument(
        "--dim", type=int, default=32, help="Initial number of Unet channels (This number will be multiplied by dim_mults)"
    )
    parser.add_argument(
        "--dim_mults",
        type=tuple[int],
        default=(1, 2, 4),
        help="Multipliers of dim for each layer. It also specifies network depth.",
    )
    parser.add_argument("--in_channels", type=int, default=3, help="Channels in image.")
    parser.add_argument("--num_resnet_groups", type=int, default=4, help="Number of resnet groups.")

    # forward diffusion parameters
    parser.add_argument("--beta_start", type=float, default=0.0001, help="Starting beta value")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Ending beta value")
    parser.add_argument("--timesteps", type=int, default=256, help="Number of diffusion timesteps")
    parser.add_argument(
        "--schedule_type", type=str, choices=["cosine", "linear", "quadratic", "sigmoid"], default="cosine", help="Schedule type."
    )

    # training hyperparamers
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--l2", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--loss_type", type=str, choices=["mse", "mae", "huber"], default="mse", help="Loss function")
    parser.add_argument("--save_ckpt_dir", type=str, default="checkpoints", help="Save checkpoint directory")
    parser.add_argument("--save_every_x_steps", type=int, default=50, help="Save model every x steps")

    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator type")
    parser.add_argument("--precision", type=int, choices=[16, 32, 64], default=32, help="Precision")
    parser.add_argument("--overfit_batches", type=int, default=0, help="Determines if overfit batch")

    # logger
    parser.add_argument("--run_name", type=str, default="run-1", help="Name of run")
    parser.add_argument("--log_every_n_steps", type=int, default=50, help="Log every n steps")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    wandb.login()

    dataloader = preprocess_dataset(args.dataset, batch_size=args.batch_size)
    train(dataloader, args)

    wandb.finish()
