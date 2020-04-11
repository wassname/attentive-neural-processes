import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from .utils import ObjectDict
from .data.smart_meter import get_smartmeter_df, SmartMeterDataSet, collate_fns
from .logger import logger
from .plot import plot_from_loader


class PL_Seq2Seq(pl.LightningModule):
    def __init__(self, hparams, loss_fn=F.mse_loss, num_workers=3, MODEL_CLS=None):
        super().__init__()
        self.hparams = ObjectDict()
        self.hparams.update(
            hparams.__dict__ if hasattr(hparams, "__dict__") else hparams
        )
        self.num_workers = num_workers
        self._model = MODEL_CLS(self.hparams)
        self._datasets = None
        self.loss_fn = loss_fn
        self.train_logs = []  # HACK
        self._dfs = None
        # TODO make label name configurable
        # TODO make data source configurable

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

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

    def test_step(self, batch, batch_idx):
        pred, losses, extra = self.forward(*batch)
        # For test use a diff loss, MSE over next 24
        # loss = losses["loss"]
        loss = F.mse_loss(pred, batch[-1], reduction='none')[:, :24].mean()
        tensorboard_logs = {"test_" + k: v for k, v in losses.items()}
        return {"test_loss": loss, "log": tensorboard_logs}

    def test_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        keys = outputs[0]["log"].keys()
        tensorboard_logs = {
            k: torch.stack([x["log"][k] for x in outputs if k in x["log"]]).mean()
            for k in keys
        }
        tensorboard_logs_str = {k: f"{v}" for k, v in tensorboard_logs.items()}

        logger.info(
            f"step {self.trainer.global_step}, {tensorboard_logs_str}"
        )
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, patience=self.hparams["patience"], verbose=True, min_lr=1e-7
        )  # note early stopping has patience 3
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
