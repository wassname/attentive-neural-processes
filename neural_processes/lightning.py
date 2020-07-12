import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from .utils import round_values, merge_dict_torch
from .utils import ObjectDict, agg_logs, round_values
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
        y_dist, losses, extra = self._model(*args, **kwargs)
        assert torch.isfinite(losses["loss"])
        return y_dist, losses, extra

    # steps
    def training_step(self, batch, batch_idx):
        y_dist, losses, extra = self.forward(*batch)
        tensorboard_logs = {"train_" + k: v for k, v in losses.items()}
        return {"loss": losses["loss"], "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        y_dist, losses, extra = self.forward(*batch)
        tensorboard_logs = {"val_" + k: v for k, v in losses.items()}
        return {"val_loss": losses["loss"], "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        y_dist, losses, extra = self.forward(*batch)
        tensorboard_logs = {"test_" + k: v for k, v in losses.items()}
        return {"test_loss": losses["loss"], "log": tensorboard_logs}

    # epoch ends
    def _epoch_end(self, outputs, name):
        outputs = [o.get("log", o) for o in outputs]
        outputs = merge_dict_torch(outputs)
        logger.info(
            f"{name} step={self.trainer.global_step}, outputs={round_values(outputs)}"
        )
        return {
            f"{name}_loss": outputs.get(f"{name}_loss"),
            "log": outputs,
        }

    def training_epoch_end(self, outputs):
        if int(self.hparams.vis_i) > 0:
            self.show_image(loader = self.train_dataloader(), title='train ')
        return self._epoch_end(outputs, "train")

    def test_epoch_end(self, outputs):
        return self._epoch_end(outputs, "test")

    def validation_epoch_end(self, outputs):
        outputs = self._epoch_end(outputs, "val")
        if int(self.hparams.vis_i) > 0:
            self.show_image(loader = self.val_dataloader(), title='val ')
        return outputs

    def show_image(self, loader, title=''):        
        # https://github.com/PytorchLightning/pytorch-lightning/blob/f8d9f8f/pytorch_lightning/core/lightning.py#L293
        vis_i = min(int(self.hparams["vis_i"]), len(loader.dataset))
        if isinstance(self.hparams["vis_i"], str):
            # if it's a string we show
            image = plot_from_loader(loader, self, i=int(vis_i), title=f'{title}, step={self.trainer.global_step}')
            plt.show()
        else:
            # if it's a int we send to tensorboard
            image = plot_from_loader_to_tensor(loader, self, i=vis_i)
            self.logger.experiment.add_image('val/image', image, self.trainer.global_step)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, patience=self.hparams["patience"], verbose=True, min_lr=1e-7
        )
        return [optim], [scheduler]

    def _get_cache_dfs(self):
        if self._dfs is None:
            df_train, df_test = get_smartmeter_df()
            self._dfs = dict(df_train=df_train, df_test=df_test)
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
                self.hparams["num_context"], self.hparams["num_extra_target"], sample=True
            ),
            sampler=sampler,
            num_workers=self.hparams["num_workers"],
        )

    @pl.data_loader
    def val_dataloader(self):
        df_test = self._get_cache_dfs()['df_test']
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
                self.hparams["num_context"], self.hparams["num_extra_target"], sample=False
            ),
            num_workers=self.hparams["num_workers"],
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
                self.hparams["num_context"], self.hparams["num_extra_target"], sample=False
            ),
            sampler=sampler,
            num_workers=self.hparams["num_workers"],
        )
