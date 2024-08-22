import math
from typing import Callable, List

import torch
import torch.nn as nn


def init_layer(layer: nn.Module):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn: nn.Module):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def init_gru(rnn: nn.GRU):
    """Initialize a GRU layer."""

    def _concat_init(tensor: torch.Tensor, init_funcs: List[Callable]):
        (length, _) = tensor.shape
        fan_in = length // len(init_funcs)

        for i, init_func in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])

    def _inner_uniform(tensor: torch.Tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, "fan_in")
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))

    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, "weight_ih_l{}".format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform],
        )
        torch.nn.init.constant_(getattr(rnn, "bias_ih_l{}".format(i)), 0)

        _concat_init(
            getattr(rnn, "weight_hh_l{}".format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_],
        )
        torch.nn.init.constant_(getattr(rnn, "bias_hh_l{}".format(i)), 0)
