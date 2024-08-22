from typing import Any, Dict, Union

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from modules.note import RegressNoteModel
from modules.pedal import RegressPedalModel
from preprocess.feature import create_feature, create_mel_transform
from training.config import Config


def binary_cross_entropy(output, target, mask):
    """Binary crossentropy (BCE) with mask. The positions where mask=0 will be
    deactivated when calculation BCE.
    """
    eps = 1e-7
    output = torch.clamp(output, eps, 1.0 - eps)
    loss = F.binary_cross_entropy(output, target, weight=mask, reduction="sum")
    return loss / torch.sum(mask)


class TranscriberModule(LightningModule):
    def __init__(
        self,
        model: Union[RegressNoteModel, RegressPedalModel],
        config: Config,
        optimizer_class: Any,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.all_loss = []
        self.val_loss_all = []
        self.epoch_loss = []

    def note_loss(self, output: Dict, target: Dict):
        onset_loss = binary_cross_entropy(
            output["reg_onset_output"], target["reg_onset_roll"], target["mask_roll"]
        )
        offset_loss = binary_cross_entropy(
            output["reg_offset_output"], target["reg_offset_roll"], target["mask_roll"]
        )
        frame_loss = binary_cross_entropy(
            output["frame_output"], target["frame_roll"], target["mask_roll"]
        )
        velocity_loss = binary_cross_entropy(
            output["velocity_output"],
            target["velocity_roll"] / 128,
            target["mask_roll"],
        )
        return onset_loss + offset_loss + frame_loss + velocity_loss

    def pedal_loss(self, output: Dict, target: Dict):
        onset_loss = F.binary_cross_entropy(
            output["reg_onset_output"], target["reg_onset_roll"][:, :, None]
        )
        offset_loss = F.binary_cross_entropy(
            output["reg_offset_output"], target["reg_offset_roll"][:, :, None]
        )
        frame_loss = F.binary_cross_entropy(
            output["frame_output"], target["frame_roll"][:, :, None]
        )
        return onset_loss + offset_loss + frame_loss

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch: torch.Tensor, _: int):
        (
            audio,
            onset_roll,
            offset_roll,
            reg_onset_roll,
            reg_offset_roll,
            frame_roll,
            velocity_roll,
            mask_roll,
            pedal_onset_roll,
            pedal_offset_roll,
            reg_pedal_onset_roll,
            reg_pedal_offset_roll,
            pedal_frame_roll,
        ) = batch
        mel_transform = create_mel_transform(self.config, self.device)
        feature = create_feature(audio, self.config, self.device, mel_transform)
        output = self.model(feature.unsqueeze(1))

        if isinstance(self.model, RegressNoteModel):
            loss = self.note_loss(
                output,
                {
                    "reg_onset_roll": reg_onset_roll,
                    "reg_offset_roll": reg_offset_roll,
                    "frame_roll": frame_roll,
                    "velocity_roll": velocity_roll,
                    "mask_roll": mask_roll,
                },
            )
        elif isinstance(self.model, RegressPedalModel):
            loss = self.pedal_loss(
                output,
                {
                    "reg_onset_roll": reg_pedal_onset_roll,
                    "reg_offset_roll": reg_pedal_offset_roll,
                    "frame_roll": pedal_frame_roll,
                },
            )
        else:
            raise ValueError("Invalid output type")

        self.all_loss.append(loss.item())
        self.epoch_loss.append(loss.item())
        self.log("loss", loss)
        return loss

    
    def validation_step(self, batch: torch.Tensor, _: int):
        (
            audio,
            onset_roll,
            offset_roll,
            reg_onset_roll,
            reg_offset_roll,
            frame_roll,
            velocity_roll,
            mask_roll,
            pedal_onset_roll,
            pedal_offset_roll,
            reg_pedal_onset_roll,
            reg_pedal_offset_roll,
            pedal_frame_roll,
        ) = batch
        mel_transform = create_mel_transform(self.config, self.device)
        feature = create_feature(audio, self.config, self.device, mel_transform)
        output = self.model(feature.unsqueeze(1))

        if isinstance(self.model, RegressNoteModel):
            loss = self.note_loss(
                output,
                {
                    "reg_onset_roll": reg_onset_roll,
                    "reg_offset_roll": reg_offset_roll,
                    "frame_roll": frame_roll,
                    "velocity_roll": velocity_roll,
                    "mask_roll": mask_roll,
                },
            )
        elif isinstance(self.model, RegressPedalModel):
            loss = self.pedal_loss(
                output,
                {
                    "reg_onset_roll": reg_pedal_onset_roll,
                    "reg_offset_roll": reg_pedal_offset_roll,
                    "frame_roll": pedal_frame_roll,
                },
            )
        else:
            raise ValueError("Invalid output type")

        self.val_loss_all.append(loss.item())
        self.log("val_loss", loss)
        return loss

    def training_epoch_start(self, _):
        self.epoch_loss = []

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer
