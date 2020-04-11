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

def collate_fn(batch, sample=None):
    return default_collate(batch)


class LSTMNet(nn.Module):

    def __init__(self, hparams, _min_std = 0.05):
        super().__init__()
        self.hparams = hparams
        self._min_std = _min_std

        self.lstm1 = nn.LSTM(
            input_size=self.hparams.input_size,
            hidden_size=self.hparams.hidden_size,
            batch_first=True,
            num_layers=self.hparams.lstm_layers,
            bidirectional=self.hparams.bidirectional,
            dropout=self.hparams.lstm_dropout,
        )
        self.hidden_out_size = (
            self.hparams.hidden_size
            * (self.hparams.bidirectional + 1)
        )
        self.mean = nn.Linear(self.hidden_out_size, 1)
        self.std = nn.Linear(self.hidden_out_size, 1)

    def forward(self, context_x, context_y, target_x, target_y=None):
        device = next(self.parameters()).device
        x = torch.cat([context_x, context_y], -1).detach()

        outputs, (h_out, _) = self.lstm1(x)
        # outputs: [B, T, num_direction * H]
        y_pred = self.mean(outputs).squeeze(2)
        log_sigma = self.std(outputs).squeeze(2)

        loss = None
        if target_y is not None:
            loss = F.mse_loss(y_pred * loss_scale, y[:, -steps:, :] * loss_scale, reduction='none') / loss_scale

            assert torch.isfinite(loss)

        return y_pred, dict(loss=loss), dict()


class LSTM_PL_STD(PL_Seq2Seq):
    def __init__(self, hparams,
        MODEL_CLS=LSTMNet, **kwargs):
        super().__init__(hparams,
        MODEL_CLS=MODEL_CLS, **kwargs)

    DEFAULT_ARGS = {'bidirectional': False, 'hidden_size_power': 4, 'learning_rate': 0.0010825329363784934, 'lstm_dropout': 0.3905792111699782, 'lstm_layers': 4}

    @staticmethod
    def add_suggest(trial):
        trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        trial.suggest_uniform("lstm_dropout", 0, 0.75)
        trial.suggest_categorical("hidden_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])    
        trial.suggest_categorical("lstm_layers", [1, 2, 4, 8])
        trial.suggest_categorical("bidirectional", [False, True])
        
        # constants
        trial._user_attrs = {
            'batch_size': 16,
            'grad_clip': 40,
            'max_nb_epochs': 200,
            'num_workers': 4,
            'num_extra_target': 24*4,
            'vis_i': '670',
            'num_context': 24*4,
            'input_size': 18,
            'input_size_decoder': 17,
            'context_in_target': True,
            'output_size': 1,
            'patience': 3,
        }
        return trial

