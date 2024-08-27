import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import init_bn, init_gru, init_layer


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, momentum):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels, momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):
        """
        Args:
          input: (batch_size, in_channels, time_steps, freq_bins)

        Outputs:
          output: (batch_size, out_channels, classes_num)
        """

        x = F.relu_(self.bn1(self.conv1(input)))
        x = F.relu_(self.bn2(self.conv2(x)))

        if pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)

        return x


class AcousticFeatureExtractor(nn.Module):
    def __init__(self, classes_num: int, midfeat: int, momentum: float):
        super(AcousticFeatureExtractor, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=48, momentum=momentum)
        self.conv_block2 = ConvBlock(in_channels=48, out_channels=64, momentum=momentum)
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=96, momentum=momentum)
        self.conv_block4 = ConvBlock(
            in_channels=96, out_channels=128, momentum=momentum
        )

        self.fc5 = nn.Linear(midfeat, 768, bias=False)
        self.bn5 = nn.BatchNorm1d(768, momentum=momentum)

        self.gru = nn.GRU(
            input_size=768,
            hidden_size=256,
            num_layers=2,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=True,
        )

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc5)
        init_bn(self.bn5)
        init_gru(self.gru)
        init_layer(self.fc)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
          input: (batch_size, channels_num, time_steps, freq_bins)

        Outputs:
          output: (batch_size, time_steps, classes_num)
        """

        x = self.conv_block1(input, pool_size=(1, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(1, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.transpose(1, 2).flatten(2)
        x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)

        (x, _) = self.gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        output = torch.sigmoid(self.fc(x))
        return output
