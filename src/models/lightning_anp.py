import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from test_tube import Experiment, HyperOptArgumentParser
from src.models.model import LatentModel
from src.data.smart_meter import collate_fns, SmartMeterDataSet, get_smartmeter_df
from src.plot import plot_from_loader_to_tensor, plot_from_loader
from src.utils import ObjectDict
from matplotlib import pyplot as plt



class LatentModelPL(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = ObjectDict()
        self.hparams.update(hparams.__dict__ if hasattr(hparams, '__dict__') else hparams)
        self.model = LatentModel(**self.hparams)
        self._dfs = None
        self.train_logs = [] 

    def forward(self, context_x, context_y, target_x, target_y):
        return self.model(context_x, context_y, target_x, target_y)

    def training_step(self, batch, batch_idx):
        assert all(torch.isfinite(d).all() for d in batch)
        context_x, context_y, target_x, target_y = batch
        y_pred, losses, extra = self.forward(context_x, context_y, target_x, target_y)
        y_std = extra['dist'].scale
        loss = losses['loss'].mean()

        tensorboard_logs = {
            "train_loss": loss,
            "train/kl": losses['loss_kl'].mean(),
            "train/std": y_std.mean(),
            "train/mse": losses['loss_mse'].mean(),
        }
        assert torch.isfinite(loss)
        self.train_logs.append(tensorboard_logs)
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        assert all(torch.isfinite(d).all() for d in batch)
        context_x, context_y, target_x, target_y = batch
        y_pred, losses, extra = self.forward(context_x, context_y, target_x, target_y)
        y_std = extra['dist'].scale
        loss = losses['loss'].mean()
        
        tensorboard_logs = {
            "val_loss": loss, # This exact key is needed for metrics
            "val/kl": losses['loss_kl'].mean(),
            "val/std": y_std.mean(),
            "val/mse": losses['loss_mse'].mean(),
        }
        return {"val_loss": loss, "log": tensorboard_logs}

    # def training_end(self, outputs):
    #     logs = self.agg_logs(outputs)
    #     tensorboard_logs_str = {k: f'{v}' for k, v in logs["log"].items()}
    #     print(f"step train {self.trainer.global_step}, {tensorboard_logs_str}")
    #     return logs

    def validation_end(self, outputs):
        if int(self.hparams["vis_i"]) > 0:
            self.show_image()
        logs = self.agg_logs(outputs)
        tensorboard_logs_str = {k: f'{v}' for k, v in logs["log"].items()}

        # agg and print self.train_logs HACK https://github.com/PyTorchLightning/pytorch-lightning/issues/100
        train_logs = self.agg_logs(self.train_logs)
        train_logs_str = {k: f"{v}" for k, v in train_logs.items()}
        self.train_logs = []
        print(f"step val {self.trainer.global_step}, {tensorboard_logs_str} {train_logs}")
        return logs

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

    def agg_logs(self, outputs):
        if isinstance(outputs, dict):
            outputs = [outputs]
        aggs = {}
        if len(outputs)>0:
            for j in outputs[0]:
                if isinstance(outputs[0][j], dict):
                    # Take mean of sub dicts
                    keys = outputs[0][j].keys()
                    aggs[j] = {k: torch.stack([x[j][k] for x in outputs if k in x[j]]).mean().item() for k in keys}
                else:
                    # Take mean of numbers
                    aggs[j] = torch.stack([x[j] for x in outputs if j in x]).mean().item()
        return aggs

        # # Log hparams with metric, doesn't work
        # # self.logger.experiment.add_hparams(self.hparams.__dict__, {"avg_val_loss": avg_loss})
        # if f"{name}_loss" in outputs[0].keys():
        #     avg_loss = torch.stack([x[f"{name}_loss"] for x in outputs]).mean()
        #     assert torch.isfinite(avg_loss)
        # else:
        #     avg_loss = 0
        # return {f"avg_{name}_loss": avg_loss, "log": tensorboard_logs, "progress_bar": {}}

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def test_end(self, *args, **kwargs):
        return self.validation_end(*args, **kwargs)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.hparams["learning_rate"], weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=self.hparams["patience"], verbose=True, min_lr=1e-7)  # note early stopping has patience 3
        return [optim], [scheduler]

    def _get_cache_dfs(self):
        if self._dfs is None:
            df_train, df_val, df_test = get_smartmeter_df()
            self._dfs = dict(df_train=df_train, df_val=df_val, df_test=df_test)
        return self._dfs

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

        trial.suggest_categorical("hidden_dim", [8*2**i for i in range(6)])
        trial.suggest_categorical("latent_dim", [8*2**i for i in range(6)])
        
        trial.suggest_int("attention_layers", 1, 4)
        trial.suggest_categorical("n_latent_encoder_layers", [1, 2, 4, 8])
        trial.suggest_categorical("n_det_encoder_layers", [1, 2, 4, 8])
        trial.suggest_categorical("n_decoder_layers", [1, 2, 4, 8])
        trial.suggest_int("num_heads", 8, 8)

        trial.suggest_uniform("dropout", 0, 0.9)
        trial.suggest_uniform("attention_dropout", 0, 0.9)

        trial.suggest_categorical(
            "latent_enc_self_attn_type", ['uniform', 'multihead', 'ptmultihead']
        )
        trial.suggest_categorical("det_enc_self_attn_type",  ['uniform', 'multihead', 'ptmultihead'])
        trial.suggest_categorical("det_enc_cross_attn_type", ['uniform', 'multihead', 'ptmultihead'])

        trial.suggest_categorical("batchnorm", [False, True])
        trial.suggest_categorical("use_self_attn", [False, True])
        trial.suggest_categorical("use_lvar", [False, True])
        trial.suggest_categorical("use_deterministic_path", [False, True])
        trial.suggest_categorical("use_rnn", [True, False])

        trial._user_attrs = {
            'batch_size': 16,
            'grad_clip': 40,
            'max_nb_epochs': 200,
            'num_workers': 4,
            'num_context': 24* 4,
            'vis_i': '670',
            'num_extra_target': 24*4,
            'x_dim': 18,
            'context_in_target': True,
            'y_dim': 1,
            'patience': 3,
            'min_std': 0.005,
        }        
        return trial


