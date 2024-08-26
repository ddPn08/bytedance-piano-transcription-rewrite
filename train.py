import os
from typing import Literal

import fire
import numpy as np
import torch
import torch.utils.data as data
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)

from modules.note import RegressNoteModel
from modules.pedal import RegressPedalModel
from training.dataset import Dataset
from training.module import TranscriberModule

torch.set_float32_matmul_precision("medium")


class MyProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items["loss"] = pl_module.all_loss[-1] if pl_module.all_loss else float("nan")
        items["all_loss_mean"] = np.mean(pl_module.all_loss or float("nan"))
        items["epoch_loss_mean"] = np.mean(pl_module.epoch_loss or float("nan"))
        items['val_loss'] = pl_module.val_loss_all[-1] if pl_module.val_loss_all else float("nan")
        return items


def main(
    dataset_dir: str = "./maestro-v3.0.0-preprocessed",
    output_dir: str = "./output",
    mode: Literal["note", "pedal"] = "note",
    accelerator: str = "gpu",
    devices: str = "0,",
    max_train_epochs: int = 100,
    precision: _PRECISION_INPUT = 32,
    batch_size: int = 1,
    num_workers: int = 1,
    seed: int = -1,
    max_pitch_shift: int = 0,
    logger: str = "none",
    logger_name: str = "training",
    logger_project: str = "piano-transcription",
):
    seed = seed if seed >= 0 else torch.initial_seed()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    dataset = Dataset(
        dataset_dir,
        split="train",
        generator=generator,
        max_pitch_shift=max_pitch_shift,
    )
    dataloader = data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    val_dataset = Dataset(
        dataset_dir,
        split="validation",
    )
    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )

    config = dataset.config

    model = (
        RegressNoteModel(config.midi.num_notes)
        if mode == "note"
        else RegressPedalModel()
    )
    module = TranscriberModule(model, config, torch.optim.Adam)

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        f.write(config.model_dump_json(indent=4))

    if logger == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(
            name=logger_name,
            project=logger_project,
        )
    else:
        logger = None

    callbacks = [
        MyProgressBar(),
        ModelCheckpoint(
            save_every_n_steps=1000,
            dirpath=checkpoint_dir,
            save_top_k=10,
            mode="min",
            monitor="val_loss",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = Trainer(
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_train_epochs,
        log_every_n_steps=1,
        callbacks=callbacks,
        precision=precision,
    )
    trainer.fit(module, dataloader, val_dataloader)


if __name__ == "__main__":
    fire.Fire(main)
