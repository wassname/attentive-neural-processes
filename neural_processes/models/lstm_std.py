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
import torchvision.transforms as transforms
from argparse import ArgumentParser
import json
import pytorch_lightning as pl
import math
from matplotlib import pyplot as plt
import torch
import io
import PIL
from torchvision.transforms import ToTensor

from neural_processes.data.smart_meter import get_smartmeter_df

from neural_processes.utils import ObjectDict
from ..lightning import PL_Seq2Seq
from torch.utils.data._utils.collate import default_collate
from ..logger import logger
from ..utils import hparams_power


class LSTMNet(nn.Module):
    def __init__(self, hparams, _min_std=0.05):
        super().__init__()
        hparams = hparams_power(hparams)
        self.hparams = hparams
        self._min_std = _min_std

        self.lstm1 = nn.LSTM(
            input_size=self.hparams.x_dim+self.hparams.y_dim,
            hidden_size=self.hparams.hidden_size,
            batch_first=True,
            num_layers=self.hparams.lstm_layers,
            bidirectional=self.hparams.bidirectional,
            dropout=self.hparams.lstm_dropout,
        )
        self.hidden_out_size = self.hparams.hidden_size * (
            self.hparams.bidirectional + 1
        )
        self.mean = nn.Linear(self.hidden_out_size, 1)
        self.std = nn.Linear(self.hidden_out_size, 1)
        self._use_lvar = 0

    def forward(self, context_x, context_y, target_x, target_y=None):
        device = next(self.parameters()).device
        target_y_fake = (
            torch.ones(context_y.shape[0], target_x.shape[1], context_y.shape[2]).float().to(device) * self.hparams.nan_value
        )
        loss_scale = 1
        context = torch.cat([context_x, context_y], -1).detach()
        target = torch.cat([target_x, target_y_fake], -1).detach()
        x = torch.cat([context, target * 1], 1).detach()

        outputs, (h_out, _) = self.lstm1(x)
        # outputs: [B, T, num_direction * H]
        steps = context_y.shape[1]
        mean = self.mean(outputs)[:, steps:, :]#.squeeze(2)
        log_sigma = self.std(outputs)[:, steps:,:]  #.squeeze(2)
        
        if self._use_lvar:
            log_sigma = torch.clamp(
                log_sigma, math.log(self._min_std), -math.log(self._min_std)
            )
            sigma = torch.exp(log_sigma)
        else:
            sigma = self._min_std + (1 - self._min_std) * F.softplus(log_sigma)
        y_dist = torch.distributions.Normal(mean, sigma)

        # Loss
        loss_mse = loss_p_weighted = loss_p = None
        if target_y is not None:
            loss_mse = F.mse_loss(mean, target_y, reduction="none")
            if self._use_lvar:
                loss_p = -log_prob_sigma(target_y, mean, log_sigma)
            else:
                loss_p = -y_dist.log_prob(target_y).mean(-1)
            if self.hparams["context_in_target"]:
                loss_p[: context_x.size(1)] /= 100
                loss_mse[: context_x.size(1)] /= 100

            # Weight loss nearer to prediction time?
            weight = (torch.arange(loss_p.shape[1]) + 1).float().to(device)[None, :]
            loss_p_weighted = loss_p / torch.sqrt(weight)  # We want to weight nearer stuff more

        y_pred = y_dist.rsample if self.training else y_dist.loc
        return (
            y_pred,
            dict(loss=loss_p.mean(), loss_p_weighted=loss_p_weighted.mean(), loss_p=loss_p.mean(), loss_mse=loss_mse.mean()),
            dict(log_sigma=log_sigma, y_dist=y_dist),
        )


class LSTM_PL_STD(PL_Seq2Seq):
    def __init__(self, hparams, MODEL_CLS=LSTMNet, **kwargs):
        super().__init__(hparams, MODEL_CLS=MODEL_CLS, **kwargs)

    DEFAULT_ARGS = {
        "bidirectional": False,
        "hidden_size_power": 5,
        "learning_rate": 0.001,
        "lstm_dropout": 0.39,
        "lstm_layers": 4,
        "bidirectional": False,
    }

    @staticmethod
    def add_suggest(trial, user_attrs={}):
        trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        trial.suggest_uniform("lstm_dropout", 0, 0.85)
        trial.suggest_discrete_uniform("hidden_size_power", 3, 9, 1)
        trial.suggest_int("lstm_layers", 1, 8)
        trial.suggest_categorical("bidirectional", [False, True])

        # constants
        user_attrs_default = {
            "batch_size": 64,
            "grad_clip": 40,
            "max_nb_epochs": 200,
            "num_workers": 4,
            "vis_i": "670",
            "x_dim": 18,
            "y_dim": 1,
            "context_in_target": False,
            "patience": 3,
            'min_std': 0.005,
            'nan_value': -99.9
        }
        [trial.set_user_attr(k, v) for k, v in user_attrs_default.items()]
        [trial.set_user_attr(k, v) for k, v in user_attrs.items()]
        return trial
