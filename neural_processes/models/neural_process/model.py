import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import math

from neural_processes.modules import BatchNormSequence, BatchMLP, Attention, LSTMBlock
from neural_processes.utils import kl_loss_var, log_prob_sigma
from neural_processes.utils import hparams_power

class LatentEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=32,
        latent_dim=32,
        self_attention_type="dot",
        n_encoder_layers=3,
        min_std=0.01,
        batchnorm=False,
        dropout=0,
        attention_dropout=0,
        use_lvar=False,
        use_self_attn=False,
        attention_layers=2,
        use_lstm=False
    ):
        super().__init__()
        # self._input_layer = nn.Linear(input_dim, hidden_dim)
        if use_lstm:
            self._encoder = LSTMBlock(input_dim, hidden_dim, batchnorm=batchnorm, dropout=dropout, num_layers=n_encoder_layers)
        else:
            self._encoder = BatchMLP(input_dim, hidden_dim, batchnorm=batchnorm, dropout=dropout, num_layers=n_encoder_layers)
        if use_self_attn:
            self._self_attention = Attention(
                hidden_dim,
                self_attention_type,
                attention_layers,
                rep="identity",
                dropout=attention_dropout,
            )
        self._penultimate_layer = nn.Linear(hidden_dim, hidden_dim)
        self._mean = nn.Linear(hidden_dim, latent_dim)
        self._log_var = nn.Linear(hidden_dim, latent_dim)
        self._min_std = min_std
        self._use_lvar = use_lvar
        self._use_lstm = use_lstm
        self._use_self_attn = use_self_attn

    def forward(self, x, y):
        encoder_input = torch.cat([x, y], dim=-1)

        # Pass final axis through MLP
        encoded = self._encoder(encoder_input)

        # Aggregator: take the mean over all points
        if self._use_self_attn:
            attention_output = self._self_attention(encoded, encoded, encoded)
            mean_repr = attention_output.mean(dim=1)
        else:
            mean_repr = encoded.mean(dim=1)

        # Have further MLP layers that map to the parameters of the Gaussian latent
        mean_repr = torch.relu(self._penultimate_layer(mean_repr))

        # Then apply further linear layers to output latent mu and log sigma
        mean = self._mean(mean_repr)
        log_var = self._log_var(mean_repr)

        if self._use_lvar:
            # Clip it in the log domain, so it can only approach self.min_std, this helps avoid mode collapase
            # 2 ways, a better but untested way using the more stable log domain, and the way from the deepmind repo
            log_var = F.logsigmoid(log_var)
            log_var = torch.clamp(log_var, np.log(self._min_std), -np.log(self._min_std))
            sigma = torch.exp(0.5 * log_var)
        else:
            sigma = self._min_std + (1 - self._min_std) * torch.sigmoid(log_var * 0.5)
        dist = torch.distributions.Normal(mean, sigma)
        return dist, log_var


class DeterministicEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        x_dim,
        hidden_dim=32,
        n_d_encoder_layers=3,
        self_attention_type="dot",
        cross_attention_type="dot",
        use_self_attn=False,
        attention_layers=2,
        batchnorm=False,
        dropout=0,
        attention_dropout=0,
        use_lstm=False,
    ):
        super().__init__()
        self._use_self_attn = use_self_attn
        # self._input_layer = nn.Linear(input_dim, hidden_dim)
        if use_lstm:
            self._d_encoder = LSTMBlock(input_dim, hidden_dim, batchnorm=batchnorm, dropout=dropout, num_layers=n_d_encoder_layers)
        else:
            self._d_encoder = BatchMLP(input_dim, hidden_dim, batchnorm=batchnorm, dropout=dropout, num_layers=n_d_encoder_layers)
        if use_self_attn:
            self._self_attention = Attention(
                hidden_dim,
                self_attention_type,
                attention_layers,
                rep="identity",
                dropout=attention_dropout,
            )
        self._cross_attention = Attention(
            hidden_dim,
            cross_attention_type,
            x_dim=x_dim,
            attention_layers=attention_layers,
        )

    def forward(self, context_x, context_y, target_x):
        # Concatenate x and y along the filter axes
        d_encoder_input = torch.cat([context_x, context_y], dim=-1)

        # Pass final axis through MLP
        d_encoded = self._d_encoder(d_encoder_input)

        if self._use_self_attn:
            d_encoded = self._self_attention(d_encoded, d_encoded, d_encoded)

        # Apply attention as mean aggregation
        h = self._cross_attention(context_x, d_encoded, target_x)

        return h


class Decoder(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        hidden_dim=32,
        latent_dim=32,
        n_decoder_layers=3,
        use_deterministic_path=True,
        min_std=0.01,
        use_lvar=False,
        batchnorm=False,
        dropout=0,
        use_lstm=False,
    ):
        super(Decoder, self).__init__()
        self._target_transform = nn.Linear(x_dim, hidden_dim)
        if use_deterministic_path:
            hidden_dim_2 = 2 * hidden_dim + latent_dim
        else:
            hidden_dim_2 = hidden_dim + latent_dim
            
        if use_lstm:
            self._decoder = LSTMBlock(hidden_dim_2, hidden_dim_2, batchnorm=batchnorm, dropout=dropout, num_layers=n_decoder_layers)
        else:
            self._decoder = BatchMLP(hidden_dim_2, hidden_dim_2, batchnorm=batchnorm, dropout=dropout, num_layers=n_decoder_layers)
        self._mean = nn.Linear(hidden_dim_2, y_dim)
        self._std = nn.Linear(hidden_dim_2, y_dim)
        self._use_deterministic_path = use_deterministic_path
        self._min_std = min_std
        self._use_lvar = use_lvar

    def forward(self, r, z, target_x):
        # concatenate target_x and representation
        x = self._target_transform(target_x)

        if self._use_deterministic_path:
            z = torch.cat([r, z], dim=-1)

        r = torch.cat([z, x], dim=-1)

        r = self._decoder(r)

        # Get the mean and the variance
        mean = self._mean(r)
        log_sigma = self._std(r)

        # Bound or clamp the variance
        if self._use_lvar:
            log_sigma = torch.clamp(log_sigma, math.log(self._min_std), -math.log(self._min_std))
            sigma = torch.exp(log_sigma)
        else:
            sigma = self._min_std + (1 - self._min_std) * F.softplus(log_sigma)

        dist = torch.distributions.Normal(mean, sigma)
        return dist, log_sigma


class NeuralProcess(nn.Module):

    @staticmethod
    def FROM_HPARAMS(hparams):
        hparams = hparams_power(hparams)
        return NeuralProcess(**hparams)
    
    def __init__(self,
                 x_dim, # features in input
                 y_dim, # number of features in output
                 hidden_dim=32, # size of hidden space
                 latent_dim=32, # size of latent space
                 latent_enc_self_attn_type="ptmultihead", # type of attention: "uniform", "dot", "multihead" "ptmultihead": see attentive neural processes paper
                 det_enc_self_attn_type="ptmultihead",
                 det_enc_cross_attn_type="ptmultihead",
                 n_latent_encoder_layers=2,
                 n_det_encoder_layers=2, # number of deterministic encoder layers
                 n_decoder_layers=2,
                 use_deterministic_path=False,
                 min_std=0.01, # To avoid collapse use a minimum standard deviation, should be much smaller than variation in labels
                 dropout=0,
                 use_self_attn=False,
                 attention_dropout=0,
                 batchnorm=False,
                 use_lvar=False, # Alternative loss calculation, may be more stable
                 attention_layers=2, 
                 use_rnn=True, # use RNN/LSTM?
                 use_lstm_le=False, # use another LSTM in latent encoder instead of MLP
                 use_lstm_de=False, # use another LSTM in determinstic encoder instead of MLP
                 use_lstm_d=False, # use another lstm in decoder instead of MLP
                 context_in_target=False,
                 **kwargs,
                ):

        super(NeuralProcess, self).__init__()

        self._use_rnn = use_rnn
        self.context_in_target = context_in_target

        # Sometimes input normalisation can be important, an initial batch norm is a nice way to ensure this
        self.norm_x = BatchNormSequence(x_dim)
        self.norm_y = BatchNormSequence(y_dim)

        if self._use_rnn:
            self._lstm_x = nn.LSTM(
                input_size=x_dim,
                hidden_size=hidden_dim,
                num_layers=attention_layers,
                dropout=dropout,
                batch_first=True
            )
            self._lstm_y = nn.LSTM(
                input_size=y_dim,
                hidden_size=hidden_dim,
                num_layers=attention_layers,
                dropout=dropout,
                batch_first=True
            )
            x_dim = hidden_dim
            y_dim2 = hidden_dim
        else:
            y_dim2 = y_dim

        self._latent_encoder = LatentEncoder(
            x_dim + y_dim2,
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
            use_lstm=use_lstm_le,
        )

        self._deterministic_encoder = DeterministicEncoder(
            input_dim=x_dim + y_dim2,
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
            use_lstm=use_lstm_de,
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
            use_lstm=use_lstm_d,
            
        )
        self._use_deterministic_path = use_deterministic_path
        self._use_lvar = use_lvar

    def forward(self, context_x, context_y, target_x, target_y=None):

        # https://stackoverflow.com/a/46772183/221742
        target_x = self.norm_x(target_x)
        context_x = self.norm_x(context_x)
        context_y = self.norm_y(context_y)

        if self._use_rnn:
            # see https://arxiv.org/abs/1910.09323 where x is substituted with h = RNN(x)
            # x need to be provided as [B, T, H]
            target_x, _ = self._lstm_x(target_x)
            context_x, _ = self._lstm_x(context_x)
            context_y, _ = self._lstm_y(context_y)
            

        dist_prior, log_var_prior = self._latent_encoder(context_x, context_y)

        if target_y is not None:
            target_y2 = self.norm_y(target_y)
            if self._use_rnn:
                target_y2, _ = self._lstm_y(target_y2)
            dist_post, log_var_post = self._latent_encoder(target_x, target_y2)
            z = dist_post.loc
        else:
            z = dist_prior.loc

        num_targets = target_x.size(1)
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
                if self.context_in_target:
                    log_p[:, :context_x.size(1)] /= 100
                loss_kl = kl_loss_var(dist_prior.loc, log_var_prior,
                                      dist_post.loc, log_var_post).mean(-1)  # [B, R].mean(-1)
            else:
                log_p = dist.log_prob(target_y).mean(-1)
                if self.context_in_target:
                    log_p[:, :context_x.size(1)] /= 100 # There's the temptation for it to fit only on context, where it knows the answer, and learn very low uncertainty. 
                loss_kl = torch.distributions.kl_divergence(
                    dist_post, dist_prior).mean(-1)  # [B, R].mean(-1)
            loss_kl = loss_kl[:, None].expand(log_p.shape)
            mse_loss = F.mse_loss(dist.loc, target_y, reduction='none')[:,:context_x.size(1)].mean()
            loss_p = -log_p.mean()
            loss = (loss_kl - log_p).mean()
            loss_kl = loss_kl.mean()
            log_p = log_p.mean()

        else:
            loss_p = None
            mse_loss = None
            loss_kl = None
            loss = None

        y_pred = dist.rsample() if self.training else dist.loc
        return y_pred, dict(loss=loss, loss_p=loss_p, loss_kl=loss_kl, loss_mse=mse_loss), dict(log_sigma=log_sigma, dist=dist)
