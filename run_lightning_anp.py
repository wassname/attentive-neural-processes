import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
from argparse import ArgumentParser
from test_tube import Experiment, HyperOptArgumentParser
from src.models.lightning_anp import LatentModelPL

def add_default_args(parser):
    """
    Specify the hyperparams for this LightningModule
    """
    # MODEL specific
    # parser = HyperOptArgumentParser(parents=[parser])
    # see https://github.com/PyTorchLightning/pytorch-lightning/blob/06242c200a/pl_examples/full_examples/imagenet/imagenet_example.py#L207
    parser.add_argument('--gpus', type=str, default=-1)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--hpc_exp_number', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                               help='evaluate model on validation set', default=False)
    return parser

def parse_args(parser, argv=None):
    # parse params
    # Set our params here, in a way compatible with cli
    argv = argv.replace('\n','').strip().split(' ')
    hyperparams = parser.parse_args(argv)

    import copy
    hparams = copy.deepcopy(hyperparams)
    for k in dir(hparams):
        if k.startswith('_'):
            continue
        v = getattr(hparams, k)
        if not isinstance(v, (int, float, str, bool, torch.Tensor)):
            delattr(hparams, k)


    return hyperparams, hparams

if __name__ == "__main__":
    parser = HyperOptArgumentParser(add_help=False)

    parser = add_default_args(parser)

    # give the module a chance to add own params
    parser = LatentModelPL.add_model_specific_args(parser)

    hyperparams, hparams = parse_args(parser)
    print(hparams)

    model = LatentModelPL(hparams)

    # most basic trainer, uses good defaults
    trainer = Trainer(
        max_epochs=hparams.max_nb_epochs,
        gpus=hparams.gpus,
        nb_gpu_nodes=hparams.nodes,
        gradient_clip_val=hparams.grad_clip,
        track_grad_norm=1,
    )
    trainer.fit(model)
