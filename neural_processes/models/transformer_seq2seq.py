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
from neural_processes.modules import BatchNormSequence, LSTMBlock, NPBlockRelu2d

from neural_processes.utils import ObjectDict
from neural_processes.lightning import PL_Seq2Seq
from ..logger import logger
from ..utils import hparams_power


class TransformerSeq2SeqNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        hparams = hparams_power(hparams)
        self.hparams = hparams
        self._min_std = hparams.min_std

        hidden_out_size = self.hparams.hidden_out_size
        y_size = self.hparams.input_size - self.hparams.input_size_decoder
        x_size = self.hparams.input_size_decoder

        # Sometimes input normalisation can be important, an initial batch norm is a nice way to ensure this https://stackoverflow.com/a/46772183/221742
        self.x_norm = BatchNormSequence(x_size, affine=False)
        self.y_norm = BatchNormSequence(y_size, affine=False)

        # TODO embedd both X's the same
        if self.hparams.get('use_lstm', False):            
            self.x_emb = LSTMBlock(x_size, x_size)
            self.y_emb = LSTMBlock(y_size, y_size)
        
        self.enc_emb = nn.Linear(self.hparams.input_size, hidden_out_size)
        self.dec_emb = nn.Linear(self.hparams.input_size_decoder, hidden_out_size)
        
        encoder_norm = nn.LayerNorm(hidden_out_size)
        layer_enc = nn.TransformerEncoderLayer(
            d_model=hidden_out_size,
            dim_feedforward=hidden_out_size*4,
            dropout=self.hparams.attention_dropout,
            nhead=self.hparams.nhead,
            # activation
        )
        self.encoder = nn.TransformerEncoder(
            layer_enc, num_layers=self.hparams.nlayers, norm=encoder_norm
        )
        
        layer_dec = nn.TransformerDecoderLayer(
            d_model=hidden_out_size,
            dim_feedforward=hidden_out_size*4,
            dropout=self.hparams.attention_dropout,
            nhead=self.hparams.nhead,
        )
        decoder_norm = nn.LayerNorm(hidden_out_size)
        self.decoder = nn.TransformerDecoder(
            layer_dec, num_layers=self.hparams.nlayers, norm=decoder_norm
        )
        self.mean = NPBlockRelu2d(hidden_out_size, self.hparams.output_size)
        self.std = NPBlockRelu2d(hidden_out_size, self.hparams.output_size)
        self._use_lvar = False
        # self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, context_x, context_y, target_x, target_y=None, mask_context=True, mask_target=True):
        device = next(self.parameters()).device

        tgt_key_padding_mask = None
        # if target_y is not None and mask_target:
        #     # Mask nan's
        #     target_mask = torch.isfinite(target_y)# & (target_y!=self.hparams.nan_value)
        #     target_y[~target_mask] = 0
        #     target_y = target_y.detach()
        #     tgt_key_padding_mask = ~target_mask.any(-1)

        src_key_padding_mask = None
        # if mask_context:
        #     # Mask nan's
        #     context_mask = torch.isfinite(context_y)# & (context_y!=self.hparams.nan_value)
        #     context_y[~context_mask] = 0
        #     context_y = context_y.detach()
        #     src_key_padding_mask = ~context_mask.any(-1)# * float('-inf')

        # Norm
        context_x = self.x_norm(context_x)
        target_x = self.x_norm(target_x)
        context_y = self.y_norm(context_y)
        # if target_y is not None:
        #     target_y = self.y_norm(target_y)

        # LSTM
        if self.hparams.get('use_lstm', False):  
            context_x = self.x_emb(context_x)
            target_x = self.x_emb(target_x)
            # Size([B, C, X]) -> Size([B, C, X])
            context_y = self.y_emb(context_y)
            # Size([B, T, Y]) -> Size([B, T, Y])

        
        # Embed
        x = torch.cat([context_x, context_y], -1)
        x = self.enc_emb(x)
        # Size([B, C, X]) -> Size([B, C, hidden_dim])
        target_x = self.dec_emb(target_x)
        # Size([B, C, T]) -> Size([B, C, hidden_dim])

        x = x.permute(1, 0, 2)  # (B,C,hidden_dim) -> (C,B,hidden_dim)
        target_x = target_x.permute(1, 0, 2) 
        # requires  (C, B, hidden_dim)
        memory = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # In transformers the memory and target_x need to be the same length. Lets use a permutation invariant agg on the context
        # Then expand it, so it's available as we decode, conditional on target_x
        # (C, B, emb_dim) -> (B, emb_dim) -> (T, B, emb_dim)
        # In transformers the memory and target_x need to be the same length. Lets use a permutation invariant agg on the context
        # Then expand it, so it's available as we decode, conditional on target_x
        memory_max = memory.max(dim=0, keepdim=True)[0].expand_as(target_x)
        memory_mean = memory.mean(dim=0, keepdim=True)[0].expand_as(target_x)
        memory_last = memory[-1:, :, :].expand_as(target_x)
        memory_all = memory_max + memory_last
        if self.hparams.agg == 'max':
            memory = memory_max
        elif self.hparams.agg == 'last':
            memory = memory_last
        elif self.hparams.agg == 'all':
            memory = memory_all
        elif self.hparams.agg == 'mean':
            memory = memory_mean
        else:
            raise Exception(f"hparams.agg should be in ['last', 'max', 'mean', 'all'] not '{self.hparams.agg}''")

        outputs = self.decoder(target_x, memory, tgt_key_padding_mask=tgt_key_padding_mask)
        
        # [T, B, emb_dim] -> [B, T, emb_dim]
        outputs = outputs.permute(1, 0, 2).contiguous()
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
        loss_mse = loss_p = loss_p_weighted = None
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
            dict(loss=loss_p.mean(), loss_p=loss_p.mean(), loss_mse=loss_mse.mean(), loss_p_weighted=loss_p_weighted.mean()),
            dict(log_sigma=log_sigma, y_dist=y_dist),
        )


class TransformerSeq2Seq_PL(PL_Seq2Seq):
    def __init__(self, hparams, MODEL_CLS=TransformerSeq2SeqNet, **kwargs):
        super().__init__(hparams, MODEL_CLS=MODEL_CLS, **kwargs)

    DEFAULT_ARGS = {
        "agg": "max",
        "attention_dropout": 0.2,
        "hidden_out_size_power": 4,
        "hidden_size_power": 5,
        "learning_rate": 0.002,
        "nhead_power": 3,
        "nlayers": 2,
        "use_lstm": False
    }

    @staticmethod
    def add_suggest(trial: optuna.Trial, user_attrs={}):
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
        # we must have nhead<==hidden_size
        # so           nhead_power.max()<==hidden_size_power.min()
        trial.suggest_discrete_uniform("hidden_size_power", 4, 10, 1)
        trial.suggest_discrete_uniform("hidden_out_size_power", 4, 9, 1)
        trial.suggest_discrete_uniform("nhead_power", 1, 4, 1)
        trial.suggest_int("nlayers", 1, 12)
        trial.suggest_categorical("use_lstm", [False, True])
        trial.suggest_categorical("agg", ['last', 'max', 'mean', 'all'])   

        user_attrs_default = {
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
            'min_std': 0.005,
        }
        [trial.set_user_attr(k, v) for k, v in user_attrs_default.items()]
        [trial.set_user_attr(k, v) for k, v in user_attrs.items()]
        return trial
