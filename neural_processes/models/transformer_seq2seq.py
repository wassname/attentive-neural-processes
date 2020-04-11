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
import optuna
from torchvision.transforms import ToTensor

from neural_processes.data.smart_meter import get_smartmeter_df
from neural_processes.modules import BatchNormSequence

from neural_processes.utils import ObjectDict
from neural_processes.lightning import PL_Seq2Seq
from ..logger import logger
from ..utils import hparams_power


class TransformerSeq2SeqNet(nn.Module):
    def __init__(self, hparams, _min_std=0.05):
        super().__init__()
        hparams = hparams_power(hparams)
        self.hparams = hparams
        self._min_std = _min_std

        hidden_out_size = self.hparams.hidden_out_size
        self.enc_norm = BatchNormSequence(self.hparams.input_size)
        self.enc_emb = nn.Linear(self.hparams.input_size, hidden_out_size)
        encoder_norm = nn.LayerNorm(hidden_out_size)
        layer_enc = nn.TransformerEncoderLayer(
            d_model=hidden_out_size,
            dim_feedforward=self.hparams.hidden_size,
            dropout=self.hparams.attention_dropout,
            nhead=self.hparams.nhead,
            # activation
        )
        self.encoder = nn.TransformerEncoder(
            layer_enc, num_layers=self.hparams.nlayers, norm=encoder_norm
        )

        self.dec_norm = BatchNormSequence(self.hparams.input_size_decoder)
        self.dec_emb = nn.Linear(self.hparams.input_size_decoder, hidden_out_size)
        layer_dec = nn.TransformerDecoderLayer(
            d_model=hidden_out_size,
            dim_feedforward=self.hparams.hidden_size,
            dropout=self.hparams.attention_dropout,
            nhead=self.hparams.nhead,
        )
        decoder_norm = nn.LayerNorm(hidden_out_size)
        self.decoder = nn.TransformerDecoder(
            layer_dec, num_layers=self.hparams.nlayers, norm=decoder_norm
        )
        self.mean = nn.Linear(hidden_out_size, self.hparams.output_size)
        self.std = nn.Linear(hidden_out_size, self.hparams.output_size)
        self._use_lvar = False
        # self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, context_x, context_y, target_x, target_y=None):
        device = next(self.parameters()).device
        x = torch.cat([context_x, context_y], -1)
        # Size([B, C, input_dim])
        x = self.enc_emb(self.enc_norm(x)).permute(1, 0, 2)
        # Size([C, B, emb_dim])
        memory = self.encoder(x)
        # Size([C, B, emb_dim])
        target_x = self.dec_emb(self.dec_norm(target_x)).permute(1, 0, 2)
        # Size([T, B, input_target_dim]) -> Size([B, T, emb_dim])

        # In transformers the memory and target_x need to be the same length. Lets use a permutation invariant agg on the context
        # Then expand it, so it's available as we decode, conditional on target_x
        memory = memory.max(dim=0, keepdim=True)[0].expand_as(target_x)

        outputs = self.decoder(target_x, memory).permute(1, 0, 2).contiguous()
        # Size([B, T, emb_dim])
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

            # Weight loss nearer to prediction time?
            weight = (torch.arange(loss_p.shape[1]) + 1).float().to(device)[None, :]
            loss_p = loss_p / torch.sqrt(weight)  # We want to weight nearer stuff more

        y_pred = y_dist.rsample if self.training else y_dist.loc
        return (
            y_pred,
            dict(loss_p=loss_p.mean(), loss_mse=loss_mse.mean()),
            dict(log_sigma=log_sigma, dist=y_dist),
        )


class TransformerSeq2Seq_PL(PL_Seq2Seq):
    def __init__(self, hparams, MODEL_CLS=TransformerSeq2SeqNet, **kwargs):
        super().__init__(hparams, MODEL_CLS=MODEL_CLS, **kwargs)

    DEFAULT_ARGS = {
        "agg": "mean",
        "attention_dropout": 0.12,
        "hidden_out_size_power": 4,
        "hidden_size_power": 7,
        "learning_rate": 0.0023,
        "nhead_power": 2,
        "nlayers": 4,
    }

    @staticmethod
    def add_suggest(trial: optuna.Trial):
        """
        Add hyperparam ranges to an optuna trial and typical user attrs.
        
        Usage:
            trial = optuna.trial.FixedTrial(
                params={         
                    'hidden_size': 128,
                }
            )
            trial = add_suggest(trial)
            trainer = pl.Trainer()
            model = LSTM_PL(dict(**trial.params, **trial.user_attrs), dataset_train,
                            dataset_test, cache_base_path, norm)
            trainer.fit(model)
        """
        trial.suggest_loguniform("learning_rate", 1e-6, 1e-2)
        trial.suggest_uniform("attention_dropout", 0, 0.75)
        trial.suggest_discrete_uniform("hidden_size_power", 2, 10, 1)
        trial.suggest_discrete_uniform("hidden_out_size_power", 2, 9, 1)
        trial.suggest_discrete_uniform("nhead_power", 1, 4, 1)
        trial.suggest_int("nlayers", 1, 12)

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
            "patience": 3,
        }
        return trial
