import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

from .utils import ObjectDict, agg_logs
from .data.smart_meter import get_smartmeter_df, SmartMeterDataSet, collate_fns
from .logger import logger
from .plot import plot_from_loader, plot_from_loader_to_tensor


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
        tensorboard_logs = {"train_" + k: v for k, v in losses.items()}
        assert torch.isfinite(tensorboard_logs["train_loss"])
        return {"loss": tensorboard_logs['train_loss'], "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        context_x, context_y, target_x, target_y = batch
        assert all(torch.isfinite(d).all() for d in batch)
        y_dist, losses, extra = self.forward(context_x, context_y, target_x, target_y)
        tensorboard_logs = {"val_" + k: v for k, v in losses.items()}
        assert torch.isfinite(tensorboard_logs["val_loss"])
        return {"val_loss": tensorboard_logs["val_loss"], "log": tensorboard_logs}

    def validation_end(self, outputs):
        if int(self.hparams["vis_i"]) > 0:
            self.show_image()

        outputs = agg_logs(outputs)

        # agg and print self.train_logs HACK https://github.com/PyTorchLightning/pytorch-lightning/issues/100
        train_outputs = agg_logs(self.train_logs)
        self.train_logs = []

        print(f"step val {self.trainer.global_step}, {outputs} {train_outputs}")

        # tensorboard_logs_str = {k: f"{v}" for k, v in tensorboard_logs.items()}
        # print(f"step {self.trainer.global_step}, {outputs}")
        return {"val_loss": outputs["val_loss"], "train_loss": train_outputs.get("train_loss", None), "log": {**train_outputs["log"], **outputs["log"]}}


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

        context_x, context_y, target_x, target_y = batch
        y_dist = extra['y_dist']

        # For test use a diff loss, log_p over next <24h, so it's a standard amount of steps
        loss = -y_dist.log_prob(target_y)[:, :24].mean()
        tensorboard_logs = {"test_" + k: v for k, v in losses.items()}
        assert torch.isfinite(loss)
        return {"test_loss": loss, "log": tensorboard_logs}

    def test_end(self, outputs):

        outputs = agg_logs(outputs)

        logger.info(
            f"step {self.trainer.global_step}, {outputs}"
        )
        return {"test_loss": outputs["test_loss"], "log": outputs["log"]}

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, patience=self.hparams["patience"], verbose=True, min_lr=1e-7
        )
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

        # I want epochs to be about 5 mins of training data. That way earlystopping etc work
        sampler = None
        max_epoch_steps = self.hparams.get("max_epoch_steps", None)
        if max_epoch_steps is not None:
            inds = np.random.choice(np.arange(len(data_train)), max_epoch_steps)
            sampler = torch.utils.data.sampler.SubsetRandomSampler(inds)
        
        return torch.utils.data.DataLoader(
            data_train,
            batch_size=self.hparams["batch_size"],
            # shuffle=True,
            collate_fn=collate_fns(
                self.hparams["num_context"], self.hparams["num_extra_target"], sample=True, context_in_target=self.hparams["context_in_target"]
            ),
            sampler=sampler,
            num_workers=self.hparams["num_workers"],
        )

    @pl.data_loader
    def val_dataloader(self):
        df_test = self._get_cache_dfs()['df_val']
        data_test = SmartMeterDataSet(
            df_test, self.hparams["num_context"], self.hparams["num_extra_target"]
        )
        sampler = None
        max_epoch_steps = self.hparams.get("max_epoch_steps", None)
        if max_epoch_steps is not None:
            sampler = torch.utils.data.sampler.SubsetRandomSampler(range(int(max_epoch_steps//10)))
        return torch.utils.data.DataLoader(
            data_test,
            batch_size=self.hparams["batch_size"],
            # shuffle=False,
            sampler=sampler,
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
        max_epoch_steps = self.hparams.get("max_epoch_steps", None)
        if max_epoch_steps is not None:
            sampler = torch.utils.data.sampler.SubsetRandomSampler(range(int(max_epoch_steps//10)))
        return torch.utils.data.DataLoader(
            data_test,
            batch_size=self.hparams["batch_size"],
            # shuffle=False,
            collate_fn=collate_fns(
                self.hparams["num_context"], self.hparams["num_extra_target"], sample=False, context_in_target=self.hparams["context_in_target"]
            ),
            sampler=sampler,
        )
