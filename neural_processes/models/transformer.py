import os
import numpy as np
import pandas as pd
import torch
import optuna
from tqdm.auto import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from neural_processes.lightning import PL_Seq2Seq
from ..logger import logger
from ..utils import hparams_power


class NetTransformer(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        hparams = hparams_power(hparams)
        self.hparams = hparams
        self._min_std = hparams.min_std

        hidden_out_size = self.hparams.hidden_out_size
        enc_x_dim = self.hparams.x_dim + self.hparams.y_dim

        # Sometimes input normalisation can be important, an initial batch norm is a nice way to ensure this https://stackoverflow.com/a/46772183/221742
        self.enc_norm = BatchNormSequence(enc_x_dim, affine=False)
        
        self.enc_emb = nn.Linear(enc_x_dim, hidden_out_size)
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
        self.mean = nn.Linear(hidden_out_size, self.hparams.y_dim)
        self.std = nn.Linear(hidden_out_size, self.hparams.y_dim)
        self._use_lvar = 0

    def forward(self, context_x, context_y, target_x, target_y=None):
        device = next(self.parameters()).device
        target_y_fake = (
            torch.ones(context_y.shape[0], target_x.shape[1], context_y.shape[2]).float().to(device) * self.hparams.nan_value
        )
        context = torch.cat([context_x, context_y], -1).detach()
        target = torch.cat([target_x, target_y_fake], -1).detach()
        x = torch.cat([context, target * 1], 1).detach()

        # Masks
        x_mask = torch.isfinite(x) & (x != self.hparams.nan_value)
        x[~x_mask] = 0
        x = x.detach()
        x_key_padding_mask = ~x_mask.any(-1)

        x = self.enc_emb(self.enc_norm(x)).permute(1, 0, 2)
        
        outputs = self.encoder(x, src_key_padding_mask=x_key_padding_mask).permute(
            1, 0, 2
        )

        # Seems to help a little, especially with extrapolating out of bounds
        steps = context_y.shape[1]
        mean = self.mean(outputs)[:, steps:, :]
        log_sigma = self.std(outputs)[:, steps:, :]

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

            # Weight loss nearer to prediction time?
            weight = (torch.arange(loss_p.shape[1]) + 1).float().to(device)[None, :]
            loss_p_weighted = loss_p / torch.sqrt(weight)  # We want to weight nearer stuff more

        y_pred = y_dist.rsample if self.training else y_dist.loc
        return (
            y_pred,
            dict(loss=loss_p.mean(), loss_p=loss_p.mean(), loss_mse=loss_mse.mean(), loss_p_weighted=loss_p_weighted.mean()),
            dict(log_sigma=log_sigma, y_dist=y_dist),
        )


class PL_Transformer(PL_Seq2Seq):
    def __init__(self, hparams, MODEL_CLS=NetTransformer, **kwargs):
        super().__init__(hparams, MODEL_CLS=MODEL_CLS, **kwargs)

    DEFAULT_ARGS = {
        "attention_dropout": 0.4,
        "hidden_out_size_power": 7.0,
        "hidden_size_power": 7.0,
        "learning_rate": 0.003,
        "nhead_power": 1.0,
        "nlayers": 2,
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
        trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        trial.suggest_uniform("attention_dropout", 0, 0.9)
        trial.suggest_discrete_uniform("hidden_size_power", 2, 10, 1)
        trial.suggest_discrete_uniform("hidden_out_size_power", 2, 9, 1)
        trial.suggest_discrete_uniform("nhead_power", 1, 4, 1)
        trial.suggest_int("nlayers", 1, 12)

        user_attrs_default = {
            "batch_size": 16,
            "grad_clip": 40,
            "max_nb_epochs": 200,
            "num_workers": 4,
            "vis_i": 670,
            "x_dim": 6,
            "y_dim": 1,
            "context_in_target": False,
            "patience": 3,
            'min_std': 0.005,
            "nan_value": -99.9,
        }
        [trial.set_user_attr(k, v) for k, v in user_attrs_default.items()]
        [trial.set_user_attr(k, v) for k, v in user_attrs.items()]
        return trial
