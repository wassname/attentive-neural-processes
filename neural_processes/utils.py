from pytorch_lightning.callbacks import EarlyStopping
from optuna.integration.pytorch_lightning import _check_pytorch_lightning_availability
from pathlib import Path
import numpy as np
import torch
import math
import torch
import optuna
from .logger import logger


def agg_dict(outputs):
    keys = outputs[0].keys()
    return {
        k: torch.stack([x[k] for x in outputs if k in x])
        .mean()
        .cpu()
        .item()
        for k in keys
    }

def agg_logs(outputs):
    """
    Aggregate a list of dicts into a single (may have sub dicts but all array are aggregated)

    outputs = [
        {'val_loss': 0.7206,
            'log': {'val_loss': 0.7206, 'val_loss_p': 0.7206,}},
        {'val_loss': 0.7047,
            'log': {'val_loss': 0.7047, 'val_loss_p': 0.7047}},
    ]
    -> {'agg_val_loss': 0.7126500010490417, 'log': {'agg_val_loss': 0.7126500010490417, 'agg_val_loss_p': 0.7126500010490417, 'agg_val_loss_kl': 2.6101499770447845e-06, 'agg_val_loss_mse': 0.17669999599456787}}
    
    """
    if isinstance(outputs, dict):
        outputs = [outputs]
    
    aggs = {}
    if len(outputs) > 0:
        for j in outputs[0].keys():
            if isinstance(outputs[0][j], dict):
                # Take mean of sub dicts
                keys = outputs[0][j].keys()
                aggs[j] = {
                    'agg_'+k: torch.stack([x[j][k] for x in outputs if k in x[j]])
                    .mean()
                    .cpu()
                    .item()
                    for k in keys
                }
            else:
                # Take mean of numbers
                aggs['agg_'+j] = (
                    torch.stack([x[j] for x in outputs if j in x]).mean().cpu().item()
                )
    return aggs

def round_values(d):
    """round values in dict to 2sf."""
    def _round(v):
        if isinstance(v, float):
            return float(f"{v:.2g}")
        elif isinstance(v, dict):
            return round_values(v)
        else:
            return v
    return {k: _round(v) for k, v in d.items()}

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
    easy way to represent (hyper)parameters.

    https://stackoverflow.com/a/50613966/221742
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)

    def copy(self, **extra_params):
        return ObjectDict(**self, **extra_params)

    @property
    def __dict__(self):
        return dict(self)


def hparams_power(hparams):
    """Some value we want to go up in powers of 2
    
    So any hyper param that ends in power will be used this way.
    """
    hparams_old = hparams.copy()
    for k in hparams_old.keys():
        if k.endswith("_power"):
            k_new = k.replace("_power", "")
            hparams[k_new] = int(2 ** hparams[k])
    logger.debug("hparams %s", hparams)
    return hparams


def log_prob_sigma(value, loc, log_scale):
    """A slightly more stable (not confirmed yet) log prob taking in log_var instead of scale.
    modified from https://github.com/pytorch/pytorch/blob/2431eac7c011afe42d4c22b8b3f46dedae65e7c0/torch/distributions/normal.py#L65
    """
    var = torch.exp(log_scale * 2)
    return (
        -((value - loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
    )


def kl_loss_var(prior_mu, log_var_prior, post_mu, log_var_post):
    """
    Analytical KLD for two gaussians, taking in log_variance instead of scale ( given variance=scale**2) for more stable gradients
    
    For version using scale see https://github.com/pytorch/pytorch/blob/master/torch/distributions/kl.py#L398
    """

    var_ratio_log = log_var_post - log_var_prior
    t1 = (post_mu - prior_mu) ** 2 / log_var_prior.exp()
    kl_div = (
        var_ratio_log.exp() 
        + t1
        - 1.0
        - var_ratio_log
    )
    kl_div = 0.5 * kl_div
    logger.warning('seems to be an error in kl_loss_var, dont use it')
    return kl_div

