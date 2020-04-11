import os
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from test_tube import Experiment, HyperOptArgumentParser
from neural_processes.data.smart_meter import (
    collate_fns,
    SmartMeterDataSet,
    get_smartmeter_df,
)
import torchvision.transforms as transforms
from neural_processes.plot import plot_from_loader_to_tensor, plot_from_loader
from argparse import ArgumentParser
import json
import pytorch_lightning as pl
import math
from matplotlib import pyplot as plt
import torch
import io
import PIL
from torchvision.transforms import ToTensor

from neural_processes.modules import BatchNormSequence
from neural_processes.data.smart_meter import get_smartmeter_df

from neural_processes.utils import ObjectDict
from neural_processes.lightning import PL_Seq2Seq
from ..logger import logger
from ..utils import hparams_power


class Seq2SeqNet(nn.Module):
    def __init__(self, hparams, _min_std=0.05):
        super().__init__()
        hparams = hparams_power(hparams)
        self.hparams = hparams
        self._min_std = _min_std

        self.norm_input = BatchNormSequence(self.hparams.input_size)
        self.encoder = nn.LSTM(
            input_size=self.hparams.input_size,
            hidden_size=self.hparams.hidden_size,
            batch_first=True,
            num_layers=self.hparams.lstm_layers,
            bidirectional=self.hparams.bidirectional,
            dropout=self.hparams.lstm_dropout,
        )
        self.multihead_attn = nn.MultiheadAttention(
            self.hparams.hidden_size, num_heads=8
        )

        self.norm_target = BatchNormSequence(self.hparams.input_size_decoder)
        self.decoder = nn.LSTM(
            input_size=self.hparams.input_size_decoder,
            hidden_size=self.hparams.hidden_size,
            batch_first=True,
            num_layers=self.hparams.lstm_layers,
            bidirectional=self.hparams.bidirectional,
            dropout=self.hparams.lstm_dropout,
        )
        self.hidden_out_size = self.hparams.hidden_size * (
            self.hparams.bidirectional + 1
        )
        self.mean = nn.Linear(self.hidden_out_size, self.hparams.output_size)
        self.std = nn.Linear(self.hidden_out_size, self.hparams.output_size)
        self._use_lvar = False

    def forward(self, context_x, context_y, target_x, target_y=None):
        x = torch.cat([context_x, context_y], -1)

        # Sometimes input normalisation can be important, an initial batch norm is a nice way to ensure this
        x = self.norm_input(x)
        target_x = self.norm_target(target_x)

        _, (h_out, cell) = self.encoder(x)
        # hidden = [batch size, n layers * n directions, hid dim]
        # cell = [batch size, n layers * n directions, hid dim]

        # context_x, d_encoded, target_x = k, v, q

        # query, key, value = target_x, context_x, d_encoded
        attn_output, _ = self.multihead_attn(
            h_out.permute(1, 0, 2), h_out.permute(1, 0, 2), h_out.permute(1, 0, 2)
        )
        h_out = attn_output.permute(1, 0, 2).contiguous()
        attn_output, _ = self.multihead_attn(
            cell.permute(1, 0, 2), cell.permute(1, 0, 2), cell.permute(1, 0, 2)
        )
        cell = attn_output.permute(1, 0, 2).contiguous()

        outputs, (_, _) = self.decoder(target_x, (h_out, cell))
        # output = [batch size, seq len, hid dim * n directions]

        # outputs: [B, T, num_direction * H]
        mean = self.mean(outputs)
        log_sigma = self.std(outputs)
        if self._use_lvar:
            log_sigma = torch.clamp(
                log_sigma, math.log(self._min_std), -math.log(self._min_std)
            )
            sigma = torch.exp(log_sigma)
        else:
            sigma = self._min_std + (1 - self._min_std) * F.softplus(log_sigma)
        y_dist = torch.distributions.Normal(mean, sigma)

        # Loss
        loss_mse = loss_p = None
        if target_y is not None:
            loss_mse = F.mse_loss(mean, target_y, reduction="none")
            if self._use_lvar:
                loss_p = -log_prob_sigma(target_y, mean, log_sigma)
            else:
                loss_p = -y_dist.log_prob(target_y).mean(-1)

            if self.hparams["context_in_target"]:
                loss_p[: context_x.size(1)] /= 100
                loss_mse[: context_x.size(1)] /= 100
            # # Don't catch loss on context window
            # mean = mean[:, self.hparams.num_context:]
            # log_sigma = log_sigma[:, self.hparams.num_context:]

        y_pred = y_dist.rsample if self.training else y_dist.loc
        return (
            y_pred,
            dict(loss_p=loss_p.mean(), loss_mse=loss_mse.mean()),
            dict(log_sigma=log_sigma, dist=y_dist),
        )


class LSTMSeq2Seq_PL(PL_Seq2Seq):
    def __init__(self, hparams, MODEL_CLS=Seq2SeqNet, **kwargs):
        super().__init__(hparams, MODEL_CLS=MODEL_CLS, **kwargs)

    DEFAULT_ARGS = {
        "agg": "mean",
        "lstm_dropout": 0.22,
        "hidden_size_power": 4.0,
        "learning_rate": 0.001,
        "lstm_layers": 4,
        'bidirectional': False
    }

    @staticmethod
    def add_suggest(trial):
        trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        trial.suggest_uniform("lstm_dropout", 0, 0.75)
        trial.suggest_discrete_uniform("hidden_size_power", 3, 9, 1)
        trial.suggest_int("lstm_layers", 1, 8)
        trial.suggest_categorical("bidirectional", [False, True])

        trial._user_attrs = {
            "batch_size": 16,
            "grad_clip": 40,
            "max_nb_epochs": 200,
            "num_workers": 4,
            "num_extra_target": 24 * 4,
            "vis_i": "670",
            "num_context": 24 * 4,
            "input_size": 18,
            "input_size_decoder": 17,
            "context_in_target": False,
            "output_size": 1,
        }
        return trial
