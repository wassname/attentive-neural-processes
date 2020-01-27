import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from argparse import ArgumentParser
import json
import pytorch_lightning as pl

from src.data.smart_meter import get_smartmeter_df

from src.utils import ObjectDict

class SequenceDfDataSet(torch.utils.data.Dataset):
    def __init__(self, df, hparams, label_names=None, train=True, transforms=None):
        super().__init__()
        self.data = df
        self.hparams = hparams
        self.label_names = label_names
        self.train = train
        self.transforms=transforms
        
    def __len__(self):
        return len(self.data) - +self.hparams.window_length - self.hparams.target_length
    
    def iloc(self, idx):
        k = idx+self.hparams.window_length+self.hparams.target_length
        j = k-self.hparams.target_length
        i = j-self.hparams.window_length
        assert i>=0
        assert idx<=len(self.data)
        x_rows = self.data.iloc[i:j].copy()
        y_rows = self.data.iloc[k].to_frame().T.copy()
#         print(i,j,k)
        
        # add seconds since start of window index
        x_rows['tstp'] = (x_rows['tstp'] - x_rows['tstp'].iloc[0]).dt.total_seconds() / 86400.0
        
        # TODO we could augment by removing and backfilling some
        return x_rows, y_rows

    def __getitem__(self, idx):
        x_rows, y_rows = self.iloc(idx)
        
#         if self.train:
#             # zero and backfill some for augmentation
#             drop_inds = np.random.randint(1, len(x_rows)-1, size=int(len(x_rows)*0.3))
#             x_rows.iloc[drop_inds] = np.nan
#             x_rows = x_rows.bfill()
            
#         print(x_rows, y_rows)
        y = y_rows[self.label_names].astype(np.float32).values
        x = x_rows.astype(np.float32).values
#         print(x, y)
        return self.transforms(x).squeeze(0).float(), self.transforms(y[:, None,])[:, 0, 0].float()
    




class LSTM_PL(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = ObjectDict()
        self.hparams.update(hparams.__dict__ if hasattr(hparams, '__dict__') else hparams)
        self.lstm1 = nn.LSTM(
            input_size=self.hparams.input_size, 
            hidden_size=self.hparams.hidden_size, 
            batch_first=True,
            num_layers=self.hparams.lstm_layers,
            bidirectional=self.hparams.bidirectional,
            dropout=self.hparams.lstm_dropout,
        )
        self.hidden_out_size = self.hparams.hidden_size * self.hparams.lstm_layers * (self.hparams.bidirectional + 1)
        self.linear = nn.Linear(self.hidden_out_size, 1)
        self._dfs = None

    def forward(self, x):
        outputs, (h_out, _) = self.lstm1(x)
        h_out = h_out.permute((1, 0, 2)).reshape((-1, self.hidden_out_size))
        return self.linear(h_out)

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        tensorboard_logs = {'val_loss': loss}
        return {'val_loss': loss, 'log': tensorboard_logs}

    def validation_end(self, outputs):
        # TODO send an image to tensroboard, like in the lighting_anp.py file
        # if self.hparams["vis_i"] > 0:
        #     self.logger.experiment.add_image('val/image', image, 

        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        keys = outputs[0]["log"].keys()
        tensorboard_logs = {k: torch.stack([x["log"][k] for x in outputs if k in x["log"]]).mean() for k in keys}
        tensorboard_logs_str = {k: f'{v}' for k, v in tensorboard_logs.items()}
        print(f"step {self.trainer.global_step}, {tensorboard_logs_str}")
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def _get_cache_dfs(self):
        if self._dfs is None:
            df_train, df_test = get_smartmeter_df()
            # self._dfs = dict(df_train=df_train[:600], df_test=df_test[:600])
            self._dfs = dict(df_train=df_train, df_test=df_test)
        return self._dfs

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        df_train = self._get_cache_dfs()['df_train']
        dset_train = SequenceDfDataSet(df_train, self.hparams, label_names=['energy(kWh/hh)'], transforms=transforms.ToTensor(), train=True)
        return DataLoader(dset_train, batch_size=self.hparams.batch_size, 
                          shuffle=True, 
                          num_workers=self.hparams.num_workers)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        df_test = self._get_cache_dfs()['df_test']
        dset_test = SequenceDfDataSet(df_test, self.hparams, label_names=['energy(kWh/hh)'], train=False, transforms=transforms.ToTensor())
        return DataLoader(dset_test, batch_size=self.hparams.batch_size, shuffle=False)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        df_test = self._get_cache_dfs()['df_test']
        dset_test = SequenceDfDataSet(df_test, self.hparams, label_names=['energy(kWh/hh)'], train=False, transforms=transforms.ToTensor())
        return DataLoader(dset_test, batch_size=self.hparams.batch_size, shuffle=False)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = HyperOptArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=0.02, type=float)
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--lstm_dropout', default=0, type=float)        
        parser.add_argument('--hidden_size', default=32, type=int)
        parser.add_argument('--input_size', default=8, type=int)
        parser.add_argument('--lstm_layers', default=4, type=int)
        parser.add_argument('--bidirectional', default=False, type=bool)

        # training specific (for this model)
        parser.add_argument('--window_length', type=int, default=12)
        parser.add_argument('--target_length', type=int, default=2)
        parser.add_argument('--max_nb_epochs', default=10, type=int)
        parser.add_argument('--num_workers', default=4, type=int)

        return parser


# dset_train = SequenceDfDataSet(df_train, hparams, transforms=transforms.ToTensor())
# dset_test = SequenceDfDataSet(df_test, hparams, train=False, transforms=transforms.ToTensor())
# dset_val = SequenceDfDataSet(df_test, hparams, train=False, transforms=transforms.ToTensor())

# model = LSTM_PL(hparams)

# # most basic trainer, uses good defaults
# trainer = Trainer(
#     max_nb_epochs=hparams.max_nb_epochs,
#     gpus=hparams.gpus,
#     nb_gpu_nodes=hparams.nodes,
# )
# trainer.fit(model)
