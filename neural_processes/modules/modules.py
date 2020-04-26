import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
# from .attention import Attention as PtAttention


class LSTMBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, dropout=0, batchnorm=False, bias=False, num_layers=1
    ):
        super().__init__()
        self._lstm = nn.LSTM(
                input_size=in_channels,
                hidden_size=out_channels,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
                bias=bias
        )

    def forward(self, x):
        return self._lstm(x)[0]


class BatchNormSequence(nn.Module):
    """Applies batch norm on features of a batch first sequence."""
    def __init__(
        self, out_channels, **kwargs
    ):
        super().__init__()
        self.norm = nn.BatchNorm1d(out_channels, **kwargs)

    def forward(self, x):
        # x.shape is (Batch, Sequence, Channels)
        # Now we want to apply batchnorm and dropout to the channels. So we put it in shape
        # (Batch, Channels, Sequence) which is what BatchNorm1d expects
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        return x.permute(0, 2, 1)

class NPBlockRelu2d(nn.Module):
    """Block for Neural Processes."""

    def __init__(
        self, in_channels, out_channels, dropout=0, batchnorm=False, bias=False
    ):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)
        self.norm = nn.BatchNorm2d(out_channels) if batchnorm else False

    def forward(self, x):
        # x.shape is (Batch, Sequence, Channels)
        # We pass a linear over it which operates on the Channels
        x = self.act(self.linear(x))

        # Now we want to apply batchnorm and dropout to the channels. So we put it in shape
        # (Batch, Channels, Sequence, None) so we can use Dropout2d & BatchNorm2d
        x = x.permute(0, 2, 1)[:, :, :, None]

        if self.norm:
            x = self.norm(x)

        x = self.dropout(x)
        return x[:, :, :, 0].permute(0, 2, 1)


class BatchMLP(nn.Module):
    """Apply MLP to the final axis of a 3D tensor (reusing already defined MLPs).

    Args:
        input: input tensor of shape [B,n,d_in].
        output_sizes: An iterable containing the output sizes of the MLP as defined 
            in `basic.Linear`.
    Returns:
        tensor of shape [B,n,d_out] where d_out=output_size
    """

    def __init__(
        self, input_size, output_size, num_layers=2, dropout=0, batchnorm=False
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.initial = NPBlockRelu2d(
            input_size, output_size, dropout=dropout, batchnorm=batchnorm
        )
        self.encoder = nn.Sequential(
            *[
                NPBlockRelu2d(
                    output_size, output_size, dropout=dropout, batchnorm=batchnorm
                )
                for _ in range(num_layers - 2)
            ]
        )
        self.final = nn.Linear(output_size, output_size)

    def forward(self, x):
        x = self.initial(x)
        x = self.encoder(x)
        return self.final(x)

