"""PyTorch Lightning `dict` logger."""

from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

class DictLogger(TensorBoardLogger):
    """PyTorch Lightning `dict` logger."""
    # see https://github.com/PyTorchLightning/pytorch-lightning/blob/50881c0b31/pytorch_lightning/logging/base.py

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = []
        
    def log_hyperparams(*args, **kwargs):
        # We will do this manually with final metrics
        pass

    def log_metrics(self, metrics, step=None):
        super().log_metrics(metrics, step=step)
        self.metrics.append(metrics)
