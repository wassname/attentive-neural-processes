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
    def __init__(self,
                 x_dim,
                 y_dim,
                 hidden_dim=32,
                 latent_dim=32,
                 latent_enc_self_attn_type="dot",
                 det_enc_self_attn_type="dot",
                 det_enc_cross_attn_type="dot",
                 n_latent_encoder_layers=3,
                 n_det_encoder_layers=3,
                 n_decoder_layers=3,
                 use_deterministic_path=True,
                 min_std=0.01,
                 dropout=0,
                 use_self_attn=False,
                 attention_dropout=0,
                 batchnorm=False,
                 use_lvar=False,
                 attention_layers=2,
                 use_rnn=False,
                 **kwargs,
                ):

        super(LatentModel, self).__init__()

        self._use_rnn = use_rnn

        if self._use_rnn:
            self._lstm = nn.LSTM(
                input_size=x_dim,
                hidden_size=hidden_dim,
                num_layers=attention_layers,
                dropout=dropout,
                batch_first=True
            )
            x_dim = hidden_dim

        self._latent_encoder = LatentEncoder(
            x_dim + y_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            self_attention_type=latent_enc_self_attn_type,
            n_encoder_layers=n_latent_encoder_layers,
            attention_layers=attention_layers,
            dropout=dropout,
            use_self_attn=use_self_attn,
            attention_dropout=attention_dropout,
            batchnorm=batchnorm,
            min_std=min_std,
            use_lvar=use_lvar,
        )

        self._deterministic_encoder = DeterministicEncoder(
            input_dim=x_dim + y_dim,
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            self_attention_type=det_enc_self_attn_type,
            cross_attention_type=det_enc_cross_attn_type,
            n_d_encoder_layers=n_det_encoder_layers,
            attention_layers=attention_layers,
            use_self_attn=use_self_attn,
            dropout=dropout,
            batchnorm=batchnorm,
            attention_dropout=attention_dropout,
        )

        self._decoder = Decoder(
            x_dim,
            y_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            batchnorm=batchnorm,
            min_std=min_std,
            use_lvar=use_lvar,
            n_decoder_layers=n_decoder_layers,
            use_deterministic_path=use_deterministic_path,
            
        )
        self._use_deterministic_path = use_deterministic_path
        self._use_lvar = use_lvar

    def forward(self, context_x, context_y, target_x, target_y=None):
        num_targets = target_x.size(1)

        if self._use_rnn:
            # see https://arxiv.org/abs/1910.09323 where x is substituted with h = RNN(x)
            # x need to be provided as [B, T, H]
            x = torch.cat([context_x, target_x], dim=1)
            # h: [B, T, num_direction * H]
            h, _ = self._lstm(x)
            context_x = h[:, :context_x.shape[1], :]
            target_x = h[:, context_x.shape[1]:, :]

        dist_prior, log_var_prior = self._latent_encoder(context_x, context_y)

        if target_y is not None:
            dist_post, log_var_post = self._latent_encoder(target_x,
                                                             target_y)
            z = dist_post.loc
        else:
            z = dist_prior.loc

        z = z.unsqueeze(1).repeat(1, num_targets, 1)  # [B, T_target, H]

        if self._use_deterministic_path:
            r = self._deterministic_encoder(context_x, context_y,
                                            target_x)  # [B, T_target, H]
        else:
            r = None

        dist, log_sigma = self._decoder(r, z, target_x)
        if target_y is not None:

            if self._use_lvar:
                log_p = log_prob_sigma(target_y, dist.loc, log_sigma).mean(-1)  # [B, T_target, Y].mean(-1)
                kl_loss = kl_loss_var(dist_prior.loc, log_var_prior,
                                      dist_post.loc, log_var_post).mean(-1)  # [B, R].mean(-1)
            else:
                log_p = dist.log_prob(target_y).mean(-1)
                kl_loss = torch.distributions.kl_divergence(
                    dist_post, dist_prior).mean(-1)
            kl_loss = kl_loss[:, None].expand(log_p.shape)
            mse_loss = F.mse_loss(dist.loc, target_y)
            loss = (kl_loss - log_p).mean()

        else:
            log_p = None
            mse_loss = None
            kl_loss = None
            loss = None

        y_pred = dist.rsample() if self.training else dist.loc
        return y_pred, kl_loss, loss, mse_loss, dist.scale
