import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from test_tube import Experiment, HyperOptArgumentParser
from .model import NeuralProcess
from neural_processes.lightning import PL_Seq2Seq
from neural_processes.utils import ObjectDict


class PL_NeuralProcess(PL_Seq2Seq):
    def __init__(self, hparams,
        MODEL_CLS=NeuralProcess.FROM_HPARAMS, **kwargs):
        super().__init__(hparams,
        MODEL_CLS=MODEL_CLS, **kwargs)

    DEFAULT_ARGS = {
        'attention_dropout': 0,
        'attention_layers': 2,
        'batchnorm': False,
        'det_enc_cross_attn_type': 'multihead',
        'det_enc_self_attn_type': 'uniform',
        'dropout': 0,
        'hidden_dim_power': 7,
        'latent_dim_power': 7,
        'latent_enc_self_attn_type': 'uniform',
        'learning_rate': 0.002,
        'n_decoder_layers': 4,
        'n_det_encoder_layers': 4,
        'n_latent_encoder_layers': 2,
        'num_heads_power': 3,
        'use_deterministic_path': True,
        'use_lvar': True,
        'use_self_attn': True,
        'use_rnn': False,
    }

    @staticmethod
    def add_suggest(trial, user_attrs={}):        
        trial.suggest_loguniform("learning_rate", 1e-6, 1e-2)
        trial.suggest_int("attention_layers", 1, 4)
        trial.suggest_discrete_uniform("num_heads_power", 2, 4, 1)

        trial.suggest_discrete_uniform(
            "hidden_dim_power", 3, 11, 1
        )
        trial.suggest_discrete_uniform(
            "latent_dim_power", 3, 11, 1
        )
        trial.suggest_int(
            "n_latent_encoder_layers", 1, 11
        )

        trial.suggest_int("n_latent_encoder_layers", 1, 12)
        trial.suggest_int("n_det_encoder_layers", 1, 12)
        trial.suggest_int("n_decoder_layers", 1, 12)

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

        user_attrs_default = {
            'batch_size': 16,
            'grad_clip': 40,
            'max_nb_epochs': 200,
            'num_workers': 4,
            'num_context': 24* 4,
            'vis_i': '670',
            'num_extra_target': 24*4,
            'x_dim': 18,
            'context_in_target': False,
            'y_dim': 1,
            'patience': 3,
            'min_std': 0.005,
        }
        [trial.set_user_attr(k, v) for k, v in user_attrs_default.items()]
        [trial.set_user_attr(k, v) for k, v in user_attrs.items()]    
        return trial


