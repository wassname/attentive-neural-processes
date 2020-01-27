import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
from argparse import ArgumentParser
from test_tube import Experiment, HyperOptArgumentParser
from src.models.lightning_anp import LatentModelPL
from run_lightning_anp import add_default_args



def main(hparams):
    if hparams.seed is not None:
        random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        torch.backends.cudnn.deterministic = True
    
    print(hparams)
    # build model
    model = LatentModelPL(hparams)
    # configure trainer
    trainer = Trainer(
#         default_save_path=hparams.save_path,
        max_epochs=hparams.max_nb_epochs,
        gpus=hparams.gpus,
        gradient_clip_val=hparams.grad_clip,
        # track_grad_norm=1,
    )
    # train model
    if hparams.evaluate:
        trainer.run_evaluation()
    else:
        trainer.fit(model)

if __name__ == "__main__":
    parser = HyperOptArgumentParser(add_help=False)

    parser = add_default_args(parser)

    # give the module a chance to add own params
    parser = LatentModelPL.add_model_specific_args(parser)

    # parse params
    hyperparams = parser.parse_args()

    # Run some trials on a single cpu. You can view them in tensorboard, and see the hyper params
    trials = hyperparams.generate_trials(3)
    for trial in trials:
        main(trial)
