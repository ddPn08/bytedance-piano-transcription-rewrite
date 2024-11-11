import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extractor import AcousticFeatureExtractor
from .utils import init_bn, init_gru, init_layer


class RegressNoteModel(nn.Module):
    def __init__(self, classes_num: int, mel_bins: int):
        super(RegressNoteModel, self).__init__()

        midfeat = 1792
        momentum = 0.01

        self.bn0 = nn.BatchNorm2d(mel_bins, momentum=momentum)

        self.frame = AcousticFeatureExtractor(classes_num, midfeat, momentum)
        self.onset = AcousticFeatureExtractor(classes_num, midfeat, momentum)
        self.offset = AcousticFeatureExtractor(classes_num, midfeat, momentum)
        self.velocity = AcousticFeatureExtractor(classes_num, midfeat, momentum)

        self.onset_gru = nn.GRU(
            input_size=88 * 2,
            hidden_size=256,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=True,
        )
        self.onset_fc = nn.Linear(512, classes_num, bias=True)

        self.frame_gru = nn.GRU(
            input_size=88 * 3,
            hidden_size=256,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=True,
        )
        self.frame_fc = nn.Linear(512, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_gru(self.onset_gru)
        init_gru(self.frame_gru)
        init_layer(self.onset_fc)
        init_layer(self.frame_fc)

    def forward(self, x: torch.Tensor):
        """
        Args:
          input: (batch_size, data_length)

        Outputs:
          output_dict: dict, {
            'reg_onset_output': (batch_size, time_steps, classes_num),
            'reg_offset_output': (batch_size, time_steps, classes_num),
            'frame_output': (batch_size, time_steps, classes_num),
            'velocity_output': (batch_size, time_steps, classes_num)
          }
        """

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        frame_output = self.frame(x)  # (batch_size, time_steps, classes_num)
        reg_onset_output = self.onset(x)  # (batch_size, time_steps, classes_num)
        reg_offset_output = self.offset(x)  # (batch_size, time_steps, classes_num)
        velocity_output = self.velocity(x)  # (batch_size, time_steps, classes_num)

        # Use velocities to condition onset regression
        x = torch.cat(
            (reg_onset_output, (reg_onset_output**0.5) * velocity_output.detach()),
            dim=2,
        )
        (x, _) = self.onset_gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        reg_onset_output = torch.sigmoid(self.onset_fc(x))
        """(batch_size, time_steps, classes_num)"""

        # Use onsets and offsets to condition frame-wise classification
        x = torch.cat(
            (frame_output, reg_onset_output.detach(), reg_offset_output.detach()), dim=2
        )
        (x, _) = self.frame_gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        frame_output = torch.sigmoid(
            self.frame_fc(x)
        )  # (batch_size, time_steps, classes_num)
        """(batch_size, time_steps, classes_num)"""

        return {
            "reg_onset_output": reg_onset_output,
            "reg_offset_output": reg_offset_output,
            "frame_output": frame_output,
            "velocity_output": velocity_output,
        }
