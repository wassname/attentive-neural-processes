from pytorch_lightning.callbacks import EarlyStopping
from optuna.integration.pytorch_lightning import _check_pytorch_lightning_availability
from pathlib import Path
import numpy as np
import torch
import optuna


def init_random_seed(seed):
    # https://pytorch.org/docs/stable/notes/randomness.html
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    


class PyTorchLightningPruningCallback(EarlyStopping):
    """Optuna PyTorch Lightning callback to prune unpromising trials.
    Example:
        Add a pruning callback which observes validation accuracy.
        .. code::
            trainer.pytorch_lightning.Trainer(
                early_stop_callback=PyTorchLightningPruningCallback(trial, monitor='avg_val_acc'))
    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``pytorch_lightning.LightningModule.training_step`` or
            ``pytorch_lightning.LightningModule.validation_end`` and the names thus depend on
            how this dictionary is formatted.
    """

    def __init__(self, trial, monitor, **kwargs):
        # type: (optuna.trial.Trial, str) -> None

        super().__init__(monitor, **kwargs)

        _check_pytorch_lightning_availability()

        self._trial = trial
        self._monitor = monitor

    def on_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        logs = trainer.callback_metrics or {}
        current_score = logs.get(self._monitor)
        if current_score is None:
            return
        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.exceptions.TrialPruned(message)


class ObjectDict(dict):
    """
    Interface similar to an argparser
    """

    def __init__(self):
        pass

    def __setattr__(self, attr, value):
        self[attr] = value
        return self[attr]

    def __getattr__(self, attr):
        if attr.startswith("_"):
            # https://stackoverflow.com/questions/10364332/how-to-pickle-python-object-derived-from-dict
            raise AttributeError
        return dict(self)[attr]

    @property
    def __dict__(self):
        return dict(self)

