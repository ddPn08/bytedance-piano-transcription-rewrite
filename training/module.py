from typing import Any, Dict, Union

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from modules.note import RegressNoteModel
from modules.pedal import RegressPedalModel
from preprocess.feature import create_feature, create_mel_transform
from training.config import Config


def masked_bce_with_logits(
    output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
):
    """Binary crossentropy (BCE) with mask. The positions where mask=0 will be
    deactivated when calculation BCE.
    """
    bce_loss = F.binary_cross_entropy_with_logits(output, target, reduction="none")
    masked = bce_loss * mask
    return masked.sum() / mask.sum()


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
        self.epoch_loss = []

    def note_loss(self, output: Dict, target: Dict):
        onset_loss = masked_bce_with_logits(
            output["reg_onset_output"], target["reg_onset_roll"], target["mask_roll"]
        )
        offset_loss = masked_bce_with_logits(
            output["reg_offset_output"], target["reg_offset_roll"], target["mask_roll"]
        )
        frame_loss = masked_bce_with_logits(
            output["frame_output"], target["frame_roll"], target["mask_roll"]
        )
        velocity_loss = masked_bce_with_logits(
            output["velocity_output"],
            target["velocity_roll"] / 128,
            target["mask_roll"],
        )
        print(f"{onset_loss=}, {offset_loss=}, {frame_loss=}, {velocity_loss=}")
        return onset_loss + offset_loss + frame_loss + velocity_loss

    def pedal_loss(self, output: Dict, target: Dict):
        onset_loss = F.binary_cross_entropy_with_logits(
            output["reg_onset_output"], target["reg_onset_roll"][:, :, None]
        )
        offset_loss = F.binary_cross_entropy_with_logits(
            output["reg_offset_output"], target["reg_offset_roll"][:, :, None]
        )
        frame_loss = F.binary_cross_entropy_with_logits(
            output["frame_output"], target["frame_roll"][:, :, None]
        )
        return onset_loss + offset_loss + frame_loss

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch: torch.Tensor, _: int):
        torch.autograd.set_detect_anomaly(True)
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
        return loss

    def training_epoch_start(self, _):
        self.epoch_loss = []

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer
