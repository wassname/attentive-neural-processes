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
from src.data.smart_meter import collate_fns, SmartMeterDataSet, get_smartmeter_df
import torchvision.transforms as transforms
from src.plot import plot_from_loader_to_tensor, plot_from_loader
from argparse import ArgumentParser
import json
import pytorch_lightning as pl
import math
from matplotlib import pyplot as plt
import torch
import io
import PIL
from torchvision.transforms import ToTensor
from src.models.modules import BatchNormSequence
from src.data.smart_meter import get_smartmeter_df

from src.utils import ObjectDict

def log_prob_sigma(value, loc, log_scale):
    """A slightly more stable (not confirmed yet) log prob taking in log_var instead of scale.
    modified from https://github.com/pytorch/pytorch/blob/2431eac7c011afe42d4c22b8b3f46dedae65e7c0/torch/distributions/normal.py#L65
    """
    var = torch.exp(log_scale * 2)
    return (
        -((value - loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
    )


class Seq2SeqNet(nn.Module):
    def __init__(self, hparams, _min_std = 0.05):
        super().__init__()
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
        self.multihead_attn = nn.MultiheadAttention(self.hparams.hidden_size, num_heads=8)

        self.norm_target = BatchNormSequence(self.hparams.input_size_decoder)
        self.decoder = nn.LSTM(
            input_size=self.hparams.input_size_decoder,
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
        attn_output, _ = self.multihead_attn(h_out.permute(1, 0, 2), h_out.permute(1, 0, 2), h_out.permute(1, 0, 2))
        h_out = attn_output.permute(1, 0, 2).contiguous()
        attn_output, _ = self.multihead_attn(cell.permute(1, 0, 2), cell.permute(1, 0, 2), cell.permute(1, 0, 2))
        cell = attn_output.permute(1, 0, 2).contiguous()

        outputs, (_, _) = self.decoder(target_x, (h_out, cell))
        # output = [batch size, seq len, hid dim * n directions]
        
        # outputs: [B, T, num_direction * H]
        mean = self.mean(outputs)
        log_sigma = self.std(outputs)
        if self._use_lvar:
            log_sigma = torch.clamp(log_sigma, math.log(self._min_std), -math.log(self._min_std))
            sigma = torch.exp(log_sigma)
        else:
            sigma = self._min_std + (1 - self._min_std) * F.softplus(log_sigma)
        y_dist=torch.distributions.Normal(mean, sigma)
        
        # Loss
        loss_mse = loss_p = None
        if target_y is not None:
            loss_mse = F.mse_loss(mean, target_y, reduction='none')
            if self._use_lvar:
                loss_p = -log_prob_sigma(target_y, mean, log_sigma)
            else:
                loss_p = -y_dist.log_prob(target_y).mean(-1)
            
            if self.hparams["context_in_target"]:
                loss_p[:context_x.size(1)] /= 100
                loss_mse[:context_x.size(1)] /= 100
            # # Don't catch loss on context window
            # mean = mean[:, self.hparams.num_context:]
            # log_sigma = log_sigma[:, self.hparams.num_context:]

        y_pred = y_dist.rsample if self.training else y_dist.loc
        return y_pred, dict(loss_p=loss_p.mean(), loss_mse=loss_mse.mean()), dict(log_sigma=log_sigma, dist=y_dist)


class LSTMSeq2Seq_PL(pl.LightningModule):
    def __init__(self, hparams):
        # TODO make label name configurable
        # TODO make data source configurable
        super().__init__()
        self.hparams = ObjectDict()
        self.hparams.update(
            hparams.__dict__ if hasattr(hparams, "__dict__") else hparams
        )
        self.model = Seq2SeqNet(self.hparams)
        self._dfs = None

    def forward(self, context_x, context_y, target_x, target_y):
        return self.model(context_x, context_y, target_x, target_y)

    def training_step(self, batch, batch_idx):
        # REQUIRED
        assert all(torch.isfinite(d).all() for d in batch)
        context_x, context_y, target_x, target_y = batch
        y_dist, losses, extra = self.forward(context_x, context_y, target_x, target_y)
        loss = losses['loss_p'] # + loss_mse
        tensorboard_logs = {
            "train/loss": loss,
            'train/loss_mse': losses['loss_mse'],
            "train/loss_p": losses['loss_p'],
            "train/sigma": torch.exp(extra['log_sigma']).mean()}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        context_x, context_y, target_x, target_y = batch
        assert all(torch.isfinite(d).all() for d in batch)
        y_dist, losses, extra = self.forward(context_x, context_y, target_x, target_y)
        loss = losses['loss_p'] # + loss_mse
        tensorboard_logs = {
            "val_loss": loss,
            'val/loss_mse': losses['loss_mse'],
            "val/loss_p": losses['loss_p'],
            "val/sigma": torch.exp(extra['log_sigma']).mean()}
        return {"val_loss": loss, "log": tensorboard_logs}

    def validation_end(self, outputs):
        if int(self.hparams["vis_i"]) > 0:
            self.show_image()

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        keys = outputs[0]["log"].keys()
        tensorboard_logs = {
            k: torch.stack([x["log"][k] for x in outputs if k in x["log"]]).mean()
            for k in keys
        }
        tensorboard_logs_str = {k: f"{v}" for k, v in tensorboard_logs.items()}
        print(f"step {self.trainer.global_step}, {tensorboard_logs_str}")
        assert torch.isfinite(avg_loss)
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}


    def show_image(self):        
        # https://github.com/PytorchLightning/pytorch-lightning/blob/f8d9f8f/pytorch_lightning/core/lightning.py#L293
        loader = self.val_dataloader()
        vis_i = min(int(self.hparams["vis_i"]), len(loader.dataset))
        # print('vis_i', vis_i)
        if isinstance(self.hparams["vis_i"], str):
            image = plot_from_loader(loader, self, i=int(vis_i))
            plt.show()
        else:
            image = plot_from_loader_to_tensor(loader, self, i=vis_i)
            self.logger.experiment.add_image('val/image', image, self.trainer.global_step)

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def test_end(self, *args, **kwargs):
        return self.validation_end(*args, **kwargs)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, patience=self.hparams["patience"], verbose=True, min_lr=1e-5
        )  # note early stopping has patient 3
        return [optim], [scheduler]

    def _get_cache_dfs(self):
        if self._dfs is None:
            df_train, df_val, df_test = get_smartmeter_df()
            self._dfs = dict(df_train=df_train, df_val=df_val, df_test=df_test)
        return self._dfs

    @pl.data_loader
    def train_dataloader(self):
        df_train = self._get_cache_dfs()['df_train']
        data_train = SmartMeterDataSet(
            df_train, self.hparams["num_context"], self.hparams["num_extra_target"]
        )
        return torch.utils.data.DataLoader(
            data_train,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            collate_fn=collate_fns(
                self.hparams["num_context"], self.hparams["num_extra_target"], sample=True, context_in_target=self.hparams["context_in_target"]
            ),
            num_workers=self.hparams["num_workers"],
        )

    @pl.data_loader
    def val_dataloader(self):
        df_test = self._get_cache_dfs()['df_val']
        data_test = SmartMeterDataSet(
            df_test, self.hparams["num_context"], self.hparams["num_extra_target"]
        )
        return torch.utils.data.DataLoader(
            data_test,
            batch_size=self.hparams["batch_size"],
            shuffle=False,
            collate_fn=collate_fns(
                self.hparams["num_context"], self.hparams["num_extra_target"], sample=False, context_in_target=self.hparams["context_in_target"]
            ),
        )

    @pl.data_loader
    def test_dataloader(self):
        df_test = self._get_cache_dfs()['df_test']
        data_test = SmartMeterDataSet(
            df_test, self.hparams["num_context"], self.hparams["num_extra_target"]
        )
        return torch.utils.data.DataLoader(
            data_test,
            batch_size=self.hparams["batch_size"],
            shuffle=False,
            collate_fn=collate_fns(
                self.hparams["num_context"], self.hparams["num_extra_target"], sample=False, context_in_target=self.hparams["context_in_target"]
            ),
        )

    @staticmethod
    def add_suggest(trial):
        trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        trial.suggest_uniform("lstm_dropout", 0, 0.75)
        trial.suggest_categorical("hidden_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])    
        trial.suggest_categorical("lstm_layers", [1, 2, 4, 8])    
        trial.suggest_categorical("bidirectional", [False, True])    
        

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
                'output_size': 1
        }
        return trial
