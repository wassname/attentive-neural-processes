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
from torch.utils.data._utils.collate import default_collate

def collate_fn(batch, sample=None):
    return default_collate(batch)

def log_prob_sigma(value, loc, log_scale):
    """A slightly more stable (not confirmed yet) log prob taking in log_var instead of scale.
    modified from https://github.com/pytorch/pytorch/blob/2431eac7c011afe42d4c22b8b3f46dedae65e7c0/torch/distributions/normal.py#L65
    """
    var = torch.exp(log_scale * 2)
    return (
        -((value - loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
    )

class SequenceDfDataSet(torch.utils.data.Dataset):
    def __init__(self, df, hparams, label_names=None, train=True, transforms=None):
        super().__init__()
        self.data = df
        self.hparams = hparams
        self.label_names = label_names
        self.train = train
        self.transforms = transforms

    def __len__(self):
        return len(self.data) - self.hparams.window_length - self.hparams.target_length - 1

    def iloc(self, idx):
        k = idx + self.hparams.window_length + self.hparams.target_length
        j = k - self.hparams.target_length
        i = j - self.hparams.window_length
        assert i >= 0
        assert idx <= len(self.data)

        x_rows = self.data.iloc[i:k].copy()
        # x_rows = x_rows.drop(columns=self.label_names)
        # Note the NP models do have access to the previous labels for the context, we will allow the LSTM to do the same. Although it will likely just return an autoregressive solution for the first half...
        x_rows.loc[x_rows.index[self.hparams.window_length:], self.label_names] = 0
        assert len(x_rows.loc[x_rows.index[self.hparams.window_length:], self.label_names])>0
        assert (x_rows.loc[x_rows.index[self.hparams.window_length:], self.label_names]==0).all().all()

        y_rows = self.data[self.label_names].iloc[i+1:k+1].copy()
        #         print(i,j,k)

        # add seconds since start of window index
        x_rows["tstp"] = (
            x_rows["tstp"] - x_rows["tstp"].iloc[0]
        ).dt.total_seconds() / 86400.0
        return x_rows, y_rows

    def __getitem__(self, idx):
        x_rows, y_rows = self.iloc(idx)

        x = x_rows.astype(np.float32).values
        y = y_rows[self.label_names].astype(np.float32).values
        return (
            self.transforms(x).squeeze(0).float(),
            self.transforms(y).squeeze(0).squeeze(-1).float(),
        )


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

    def forward(self, x):
        outputs, (h_out, _) = self.lstm1(x)
        # outputs: [B, T, num_direction * H]
        mean = self.mean(outputs).squeeze(2)
        log_sigma = self.std(outputs).squeeze(2)
        # log_sigma = torch.clamp(log_sigma, math.log(self._min_std), -math.log(self._min_std))
        return mean, log_sigma


class LSTM_PL_STD(pl.LightningModule):
    def __init__(self, hparams):
        # TODO make label name configurable
        # TODO make data source configurable
        super().__init__()
        self.hparams = ObjectDict()
        self.hparams.update(
            hparams.__dict__ if hasattr(hparams, "__dict__") else hparams
        )
        self._model = LSTMNet(self.hparams)
        self._dfs = None
        self._use_lvar = False
        self._min_std = 0.005

        self.default_args = {'bidirectional': False, 'hidden_size_power': 4, 'learning_rate': 0.0010825329363784934, 'lstm_dropout': 0.3905792111699782, 'lstm_layers': 4}

    def forward(self, x):
        return self._model(x)

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        mean, log_sigma = self.forward(x)

        # Don't catch loss on context window
        mean = mean[:, self.hparams.window_length:]
        log_sigma = log_sigma[:, self.hparams.window_length:]
        if self._use_lvar:
            log_sigma = torch.clamp(log_sigma, math.log(self._min_std), -math.log(self._min_std))
            sigma = torch.exp(log_sigma)
        else:
            sigma = self._min_std + (1 - self._min_std) * F.softplus(log_sigma)
        y_dist = torch.distributions.Normal(mean, sigma)

        y = y[:, self.hparams.window_length:]

        loss_mse = F.mse_loss(mean, y).mean()
        if self._use_lvar:
            loss_p = -log_prob_sigma(y, mean, log_sigma).mean()
        else:
            loss_p = -y_dist.log_prob(y).mean()
        loss = loss_p # + loss_mse
        tensorboard_logs = {"train/loss": loss, 'train/loss_mse': loss_mse, "train/loss_p": loss_p, "train/sigma": sigma.mean()}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mean, log_sigma = self.forward(x)

        # Don't catch loss on context window
        mean = mean[:, self.hparams.window_length:]
        log_sigma = log_sigma[:, self.hparams.window_length:]

        sigma = torch.exp(log_sigma)
        y_dist = torch.distributions.Normal(mean, sigma)

        y = y[:, self.hparams.window_length:]

        loss_mse = F.mse_loss(mean, y).mean()
        loss_p = -log_prob_sigma(y, mean, log_sigma).mean()
        loss = loss_p # + loss_mse
        tensorboard_logs = {"val_loss": loss, 'val/loss':loss, 'val/loss_mse': loss_mse, "val/loss_p": loss_p, "val/sigma": sigma.mean()}
        return {"val_loss": loss, "log": tensorboard_logs}

    def validation_end(self, outputs):
        # TODO send an image to tensroboard, like in the lighting_anp.py file
        if int(self.hparams["vis_i"]) > 0:
            loader = self.val_dataloader()
            vis_i = min(int(self.hparams["vis_i"]), len(loader.dataset))
        if isinstance(self.hparams["vis_i"], str):
            image = plot_from_loader(loader, self, i=vis_i)
            plt.show()
        else:
            image = plot_from_loader_to_tensor(loader, self, i=vis_i)
            self.logger.experiment.add_image(
                "val/image", image, self.trainer.global_step
            )

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

    def train_dataloader(self):
        df_train = self._get_cache_dfs()["df_train"]
        dset_train = SequenceDfDataSet(
            df_train,
            self.hparams,
            label_names=["energy(kWh/hh)"],
            transforms=transforms.ToTensor(),
            train=True,
        )
        return DataLoader(
            dset_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        df_test = self._get_cache_dfs()["df_val"]
        dset_test = SequenceDfDataSet(
            df_test,
            self.hparams,
            label_names=["energy(kWh/hh)"],
            train=False,
            
            transforms=transforms.ToTensor(),
        )
        return DataLoader(dset_test, batch_size=self.hparams.batch_size, shuffle=False,collate_fn=collate_fn,)

    @pl.data_loader
    def test_dataloader(self):
        df_test = self._get_cache_dfs()["df_test"]
        dset_test = SequenceDfDataSet(
            df_test,
            self.hparams,
            label_names=["energy(kWh/hh)"],
            train=False,
            
            transforms=transforms.ToTensor(),
        )
        return DataLoader(dset_test, batch_size=self.hparams.batch_size, shuffle=False, collate_fn=collate_fn,)

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


def plot_from_loader(loader, model, title='', i=670, n=1, window_len=0):
    dset_test = loader.dataset
    label_names = dset_test.label_names
    y_trues = []
    y_preds = []
    vis_i = min(i, len(dset_test))
    for i in tqdm(range(vis_i, vis_i + n)):
        x_rows, y_rows = dset_test.iloc(i)
        x, y = dset_test[i]
        device = next(model.parameters()).device
        x = x[None, :].to(device)
        model.eval()
        with torch.no_grad():
            y_hat, log_sigma = model.forward(x)
            y_hat = y_hat.cpu().squeeze(0).numpy()
            sigma = log_sigma.exp().cpu().squeeze(0).numpy()

        dt = y_rows.iloc[0].name

        y_hat_rows = y_rows.copy()
        y_hat_rows[label_names[0]] = y_hat
        y_hat_rows['sigma'] = sigma
        y_trues.append(y_rows)
        y_preds.append(y_hat_rows)

    df_trues = pd.concat(y_trues)
    df_preds = pd.concat(y_preds)

    plt.figure()
    df_trues[label_names[0]].plot(label="y_true", style="k:")
    ylims = plt.ylim()
    df_preds[label_names[0]][window_len:].plot(label="y_pred", style="b", linewidth=2,)

    std = df_preds['sigma'][window_len:]
    mean = df_preds[label_names[0]][window_len:]
    plt.fill_between(
        df_preds.index[window_len:],
        mean - std,
        mean + std,
        alpha=0.25,
        facecolor="blue",
        interpolate=True,
        label="uncertainty",
    )
    plt.fill_between(
        df_preds.index[window_len:],
        mean - std * 2,
        mean + std * 2,
        alpha=0.125,
        facecolor="blue",
        interpolate=True,
        label="uncertainty",
    )
    plt.legend()
    t_ahead = pd.Timedelta("30T") * model.hparams.target_length
    plt.title(f"predicting {t_ahead} ahead. {title}")
    plt.ylim(*ylims)
    # plt.show()


def plot_from_loader_to_tensor(*args, **kwargs):
    plot_from_loader(*args, **kwargs)

    # Send fig to tensorboard
    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg")
    plt.close()
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)  # .unsqueeze(0)
    return image
