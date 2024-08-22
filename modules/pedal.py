import torch
import torch.nn as nn

from .feature_extractor import AcousticFeatureExtractor
from .utils import init_bn


class RegressPedalModel(nn.Module):
    def __init__(self):
        super(RegressPedalModel, self).__init__()

        mel_bins = 229

        midfeat = 1792
        momentum = 0.01

        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)

        self.reg_pedal_onset_model = AcousticFeatureExtractor(1, midfeat, momentum)
        self.reg_pedal_offset_model = AcousticFeatureExtractor(1, midfeat, momentum)
        self.reg_pedal_frame_model = AcousticFeatureExtractor(1, midfeat, momentum)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)

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

        # x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        # x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        reg_onset_output = self.reg_pedal_onset_model(
            x
        )  # (batch_size, time_steps, classes_num)
        reg_offset_output = self.reg_pedal_offset_model(
            x
        )  # (batch_size, time_steps, classes_num)
        frame_output = self.reg_pedal_frame_model(
            x
        )  # (batch_size, time_steps, classes_num)

        return {
            "reg_onset_output": reg_onset_output,
            "reg_offset_output": reg_offset_output,
            "frame_output": frame_output,
        }
