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

    def forward(self, context_x, context_y, target_x, target_y):
        return self.model(context_x, context_y, target_x, target_y)

    def training_step(self, batch, batch_idx):
        assert all(torch.isfinite(d).all() for d in batch)
        context_x, context_y, target_x, target_y = batch
        y_pred, losses, extra =  = self.forward(context_x, context_y, target_x, target_y)
        y_std = extra['dist'].scale

        tensorboard_logs = {
            "train_loss": losses['loss'],
            "train/kl": losses['loss_kl'].mean(),
            "train/std": losses['y_std'].mean(),
            "train/mse": losses['loss_mse'].mean(),
        }
        assert torch.isfinite(loss)
        # print('device', next(self.model.parameters()).device)
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        assert all(torch.isfinite(d).all() for d in batch)
        context_x, context_y, target_x, target_y = batch
        y_pred, losses, extra = self.forward(context_x, context_y, target_x, target_y)
        y_std = extra['dist'].scale
        
        tensorboard_logs = {
            "val_loss": losses['loss'], # This exact key is needed for metrics
            "val/kl": losses['loss_kl'].mean(),
            "val/mse": losses['loss_mse'].mean(),
            "val/std": losses['y_std'].mean(),
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
        print(f"step val {self.trainer.global_step}, {tensorboard_logs_str}")
        return logs

    def show_image(self):        
        # https://github.com/PytorchLightning/pytorch-lightning/blob/f8d9f8f/pytorch_lightning/core/lightning.py#L293
        loader = self.val_dataloader()[0]
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
        for j in outputs[0]:
            if isinstance(outputs[0][j], dict):
                # Take mean of sub dicts
                keys = outputs[0][j].keys()
                aggs[j] = {k: torch.stack([x[j][k] for x in outputs if k in x[j]]).mean() for k in keys}
            else:
                # Take mean of numbers
                aggs[j] = torch.stack([x[j] for x in outputs if j in x]).mean()
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
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"], weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1, verbose=True, min_lr=1e-7)  # note early stopping has patience 3
        return [optim], [scheduler]

    def _get_cache_dfs(self):
        if self._dfs is None:
            df_train, df_test = get_smartmeter_df()
            # self._dfs = dict(df_train=df_train[:600], df_test=df_test[:600])
            self._dfs = dict(df_train=df_train, df_test=df_test)
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
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser], add_help=False)
        parser.opt_range("--learning_rate", default=1e-3, type=float, tunable=True, high=1e-2, low=1e-5, log_base=10)
        
        parser.opt_list("--hidden_dim", default=128, type=int, tunable=True, options=[8*2**i for i in range(8)])
        parser.opt_list("--latent_dim", default=128, type=int, tunable=True, options=[8*2**i for i in range(8)])
        parser.add_argument("--num_heads", default=8, type=int)
        parser.add_argument("--attention_layers", default=1, type=int)
        parser.opt_list("--n_latent_encoder_layers", default=4, type=int, tunable=True, options=[1, 2, 4, 8, 16])
        parser.opt_list("--n_det_encoder_layers", default=4, type=int, tunable=True, options=[1, 2, 4, 8, 16])
        parser.opt_list("--n_decoder_layers", default=2, type=int, tunable=True, options=[1, 2, 4, 8, 16])

        parser.opt_range("--dropout", default=0, type=float, tunable=True, low=0, high=0.75)
        parser.opt_range("--attention_dropout", default=0, type=float, tunable=True, low=0, high=0.75)
        parser.add_argument("--min_std", default=0.005, type=float)

        parser.opt_list(
            "--latent_enc_self_attn_type", default="multihead", type=str, tunable=True, options=['uniform', 'dot', 'multihead', 'ptmultihead']
        )
        parser.opt_list("--det_enc_self_attn_type", default="multihead", type=str, tunable=True, options=['uniform', 'dot', 'multihead', 'ptmultihead'])
        parser.opt_list("--det_enc_cross_attn_type", default="multihead", type=str, tunable=True, options=['uniform', 'dot', 'multihead', 'ptmultihead'])

        parser.opt_list("--use_lvar", default=False, type=bool, tunable=True, options=[False, True])
        parser.opt_list("--use_rnn", default=False, type=bool, tunable=True, options=[False, True])
        parser.opt_list("--use_deterministic_path", default=True, tunable=True, type=bool, options=[False, True])
        parser.opt_list("--use_self_attn", default=True, tunable=True, type=bool, options=[False, True])
        parser.opt_list("--batchnorm", default=True, tunable=True, type=bool, options=[False, True])
        
        # training specific (for this model)
        parser.add_argument("--context_in_target", default=True, type=bool)
        parser.add_argument("--grad_clip", default=0, type=float)
        parser.add_argument("--num_context", type=int, default=24 * 2)
        parser.add_argument("--num_extra_target", type=int, default=24)
        parser.add_argument("--max_nb_epochs", default=20, type=int)
        parser.add_argument("--num_workers", default=4, type=int)

        parser.add_argument("--batch_size", default=16, type=int)
        parser.add_argument("--x_dim", default=16, type=int)
        parser.add_argument("--y_dim", default=1, type=int)
        parser.add_argument("--vis_i", default=670, type=int)
        return parser

