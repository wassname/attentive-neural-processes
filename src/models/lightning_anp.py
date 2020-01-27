import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from test_tube import Experiment, HyperOptArgumentParser
from src.models.model import LatentModel
from src.data.smart_meter import collate_fns, SmartMeterDataSet, get_smartmeter_df
from src.plot import plot_from_loader_to_tensor
from src.utils import ObjectDict


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
        y_pred, kl, loss, y_std = self.forward(context_x, context_y, target_x, target_y)
        tensorboard_logs = {
            "train/loss": loss,
            "train/kl": kl.mean(),
            "train/std": y_std.mean(),
            "train/mse": F.mse_loss(y_pred, target_y).mean(),
        }
        # print('device', next(self.model.parameters()).device)
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        assert all(torch.isfinite(d).all() for d in batch)
        context_x, context_y, target_x, target_y = batch
        y_pred, kl, loss, y_std = self.forward(context_x, context_y, target_x, target_y)
        
        tensorboard_logs = {
            "val_loss": loss,
            "val/kl": kl.mean(),
            "val/std": y_std.mean(),
            "val/mse": F.mse_loss(y_pred, target_y).mean(),
        }
        return {"val_loss": loss, "log": tensorboard_logs}

    def validation_end(self, outputs):
        if self.hparams["vis_i"] > 0:
            # https://github.com/PytorchLightning/pytorch-lightning/blob/f8d9f8f/pytorch_lightning/core/lightning.py#L293
            loader = self.val_dataloader()[0]
            vis_i = min(self.hparams["vis_i"], len(loader.dataset))
            # print('vis_i', vis_i)
            image = plot_from_loader_to_tensor(loader, self.model, i=vis_i)
            self.logger.experiment.add_image('val/image', image, self.trainer.global_step)
        
        keys = outputs[0]["log"].keys()
        # tensorboard_logs = {}
        # for k in keys:
        #     tensorboard_logs[k] = torch.stack([x["log"][k] for x in outputs if k in x["log"]]).mean()
        tensorboard_logs = {k: torch.stack([x["log"][k] for x in outputs if k in x["log"]]).mean() for k in keys}

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs_str = {k: f'{v}' for k, v in tensorboard_logs.items()}
        print(f"step {self.trainer.global_step}, {tensorboard_logs_str}")

        # Log hparams with metric, doesn't work
        # self.logger.experiment.add_hparams(self.hparams.__dict__, {"avg_val_loss": avg_loss})

        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=2, verbose=True, min_lr=1e-5) # note early stopping has patient 3
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
                self.hparams["num_context"], self.hparams["num_extra_target"], sample=True
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
                self.hparams["num_context"], self.hparams["num_extra_target"], sample=False
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
                self.hparams["num_context"], self.hparams["num_extra_target"], sample=False
            ),
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser], add_help=False)
        parser.opt_range("--learning_rate", default=1e-4, type=float, tunable=True, high=1e-2, low=1e-5, log_base=10)
        parser.add_argument("--batch_size", default=16, type=int)

        parser.add_argument("--x_dim", default=16, type=int)
        parser.add_argument("--y_dim", default=1, type=int)
        parser.add_argument("--vis_i", default=670, type=int)

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
        parser.opt_list("--use_deterministic_path", default=True, tunable=True, type=bool, options=[False, True])

        # training specific (for this model)
        parser.add_argument("--grad_clip", default=0, type=float)
        parser.add_argument("--num_context", type=int, default=24 * 2)
        parser.add_argument("--num_extra_target", type=int, default=24)
        parser.add_argument("--max_nb_epochs", default=20, type=int)
        parser.add_argument("--num_workers", default=4, type=int)
        return parser

