import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import math

from src.models.modules import LatentEncoder, DeterministicEncoder, Decoder


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
    kl_div = (
        (var_ratio_log.exp() + (post_mu - prior_mu) ** 2) / log_var_prior.exp()
        - 1.0
        - var_ratio_log
    )
    kl_div = 0.5 * kl_div
    return kl_div


class LatentModel(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        hidden_dim=32,
        latent_dim=32,
        latent_enc_self_attn_type="multihead",
        det_enc_self_attn_type="multihead",
        det_enc_cross_attn_type="multihead",
        n_latent_encoder_layers=3,
        n_det_encoder_layers=3,
        n_decoder_layers=3,
        num_heads=8,
        dropout=0,
        attention_dropout=0,
        min_std=0.1,
        use_lvar=False,
        use_deterministic_path=True,
        **kwargs
    ):

        super().__init__()
        self.use_lvar = use_lvar
        self.use_deterministic_path = use_deterministic_path

        self._latent_encoder = LatentEncoder(
            x_dim + y_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            self_attention_type=latent_enc_self_attn_type,
            n_encoder_layers=n_latent_encoder_layers,
            dropout=dropout,
            attention_dropout=attention_dropout,
            n_heads=num_heads,
            min_std=min_std,
            use_lvar=use_lvar,
        )

        self._deterministic_encoder = DeterministicEncoder(
            x_dim + y_dim,
            x_dim,
            hidden_dim=hidden_dim,
            self_attention_type=det_enc_self_attn_type,
            cross_attention_type=det_enc_cross_attn_type,
            n_d_encoder_layers=n_det_encoder_layers,
            dropout=dropout,
            attention_dropout=attention_dropout,
            n_heads=num_heads,
        )

        self._decoder = Decoder(
            x_dim,
            y_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_decoder_layers=n_decoder_layers,
            dropout=dropout,
            min_std=min_std,
            use_lvar=use_lvar,
            use_deterministic_path=use_deterministic_path
        )

    def forward(self, context_x, context_y, target_x, target_y=None):
        num_targets = target_x.size(1)

        dist_prior, log_var_prior = self._latent_encoder(context_x, context_y)

        if (target_y is not None):
            dist_post, log_var_post = self._latent_encoder(target_x, target_y)
            if self.training:
                z = dist_post.rsample()
            else:
                # instead of sampling, in test mode take the mean, this will make it more deterministic
                z = dist_post.loc
        else:
            z = dist_prior.loc

        z = z.unsqueeze(1).repeat(1, num_targets, 1)  # [B, T_target, H]

        if self.use_deterministic_path:
            r = self._deterministic_encoder(
                context_x, context_y, target_x
            )  # [B, T_target, H]
        else:
            r = None
        dist, log_sigma = self._decoder(r, z, target_x)

        if target_y is not None:
            if self.use_lvar:
                # Log likelihood has shape (batch_size, num_target, y_dim).
                # log_p = log_prob_sigma(target_y, dist.loc, log_sigma).mean(-1)
                log_p = dist.log_prob(target_y).mean(-1)
                #  KL has shape (batch_size, r_dim)
                kl_loss = kl_loss_var(
                    dist_prior.loc, log_var_prior, dist_post.loc, log_var_post
                ).mean(-1)
            else:
                log_p = dist.log_prob(target_y).mean(-1)
                kl_loss = torch.distributions.kl_divergence(dist_post, dist_prior).mean(-1)
            kl_loss = kl_loss[:, None].expand(log_p.shape)
            loss = (kl_loss - log_p).mean()

        else:
            log_p = None
            kl_loss = None
            loss = None

        y_pred = dist.rsample() if self.training else dist.loc
        return y_pred, kl_loss, loss, dist.scale

