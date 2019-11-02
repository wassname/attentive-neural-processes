import torch
from torch import nn
import torch.nn.functional as F
import math

class NPBlockRelu2d(nn.Module):
    """Block for Neural Processes."""

    def __init__(self, in_channels, out_channels, dropout=0, norm=True):
        super(NPBlockRelu2d, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)
        self.norm = nn.BatchNorm2d(out_channels) if norm else False

    def forward(self, x):
        # x.shape is (Batch, Sequence, Channels)
        # We pass a linear over it which operates on the Channels
        x = self.act(self.linear(x))

        # Now we want to apply batchnorm and dropout to the channels. So we put it in shape
        # (Batch, Channels, Sequence, None) so we can use Dropout2d
        x = x.permute(0, 2, 1)[:, :, :, None]

        if self.norm:
            x = self.norm(x)

        x = self.dropout(x)
        return x[:, :, :, 0].permute(0, 2, 1)


def block_relu(in_dim, out_dim, dropout=0, inplace=False):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(inplace=inplace),
        nn.BatchNorm1d(out_dim),
        nn.Dropout(dropout, inplace=inplace),
    )


class Attention(nn.Module):
    def __init__(self, hidden_dim, attention_type, n_heads=8, dropout=0):
        super().__init__()
        if attention_type == "uniform":
            self._attention_func = self._uniform_attention
        elif attention_type == "laplace":
            self._attention_func = self._laplace_attention
        elif attention_type == "dot":
            self._attention_func = self._dot_attention
        elif attention_type == "multihead":
            self._mattn = torch.nn.MultiheadAttention(
                hidden_dim, n_heads, bias=False, dropout=dropout
            )
            self._attention_func = self._pytorch_multihead_attention
            self.n_heads = n_heads
        else:
            raise NotImplementedError

    def forward(self, k, v, q):
        rep = self._attention_func(k, v, q)
        return rep

    def _uniform_attention(self, k, v, q):
        total_points = q.shape[1]
        rep = torch.mean(v, dim=1, keepdim=True)
        rep = rep.repeat(1, total_points, 1)
        return rep

    def _laplace_attention(self, k, v, q, scale=0.5):
        k_ = k.unsqueeze(1)
        v_ = v.unsqueeze(2)
        unnorm_weights = torch.abs((k_ - v_) * scale)
        unnorm_weights = unnorm_weights.sum(dim=-1)
        weights = torch.softmax(unnorm_weights, dim=-1)
        rep = torch.einsum("bik,bkj->bij", weights, v)
        return rep

    def _dot_attention(self, k, v, q):
        scale = q.shape[-1] ** 0.5
        unnorm_weights = torch.einsum("bjk,bik->bij", k, q) / scale
        weights = torch.softmax(unnorm_weights, dim=-1)

        rep = torch.einsum("bik,bkj->bij", weights, v)
        return rep

    def _pytorch_multihead_attention(self, k, v, q):
        # Pytorch multiheaded attention takes inputs if diff order and permutation
        o = self._mattn(q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2))[0]
        return o.permute(1, 0, 2)


class LatentEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=32,
        latent_dim=32,
        n_heads=8,
        self_attention_type="multihead",
        n_encoder_layers=3,
        min_std=0.1,
        dropout=0,
    ):
        super(LatentEncoder, self).__init__()
        self._input_layer = NPBlockRelu2d(input_dim, hidden_dim, dropout)
        self._encoder = nn.Sequential(
            *[
                NPBlockRelu2d(hidden_dim, hidden_dim, dropout)
                for _ in range(n_encoder_layers)
            ]
        )
        self._self_attention = Attention(
            hidden_dim, self_attention_type, n_heads=n_heads, dropout=dropout
        )
        self._penultimate_layer = block_relu(hidden_dim, hidden_dim, dropout)
        self._mean = nn.Linear(hidden_dim, latent_dim)
        self._log_var = nn.Linear(hidden_dim, latent_dim)
        self.min_std = min_std

    def forward(self, x, y):
        encoder_input = torch.cat([x, y], dim=-1)

        encoded = self._input_layer(encoder_input)

        encoded = self._encoder(encoded)

        attention_output = self._self_attention(encoded, encoded, encoded)

        mean_repr = attention_output.mean(dim=1)

        mean_repr = torch.relu(self._penultimate_layer(mean_repr))

        mean = self._mean(mean_repr)
        log_var = self._log_var(mean_repr)

        # Clip it in the log domain, so it can only approach self.min_std, this helps aboid mode collaprse
        log_var = F.softplus(log_var) + math.log(self.min_std)

        sigma = torch.exp(0.5 * log_var)
        dist = torch.distributions.Normal(mean, sigma)
        return dist, log_var


class DeterministicEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        x_dim,
        hidden_dim=32,
        n_d_encoder_layers=3,
        self_attention_type="multihead",
        cross_attention_type="multihead",
        dropout=0,
        n_heads=8,
    ):
        super(DeterministicEncoder, self).__init__()
        self._input_layer = NPBlockRelu2d(input_dim, hidden_dim, dropout)
        self._d_encoder = nn.Sequential(
            *[
                NPBlockRelu2d(hidden_dim, hidden_dim, dropout)
                for _ in range(n_d_encoder_layers)
            ]
        )
        self._self_attention = Attention(
            hidden_dim, self_attention_type, dropout=dropout, n_heads=n_heads
        )
        self._cross_attention = Attention(
            hidden_dim, cross_attention_type, dropout=dropout, n_heads=n_heads
        )
        self._target_transform = nn.Linear(x_dim, hidden_dim)
        self._context_transform = nn.Linear(x_dim, hidden_dim)

    def forward(self, context_x, context_y, target_x):
        d_encoder_input = torch.cat([context_x, context_y], dim=-1)
        d_encoded = self._input_layer(d_encoder_input)
        d_encoded = self._d_encoder(d_encoded)
        attention_output = self._self_attention(d_encoded, d_encoded, d_encoded)

        q = self._target_transform(target_x)
        k = self._context_transform(context_x)
        q = self._cross_attention(k, d_encoded, q)

        return q


class Decoder(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        hidden_dim=32,
        latent_dim=32,
        n_decoder_layers=3,
        min_std=0.1,
        dropout=0,
    ):
        super(Decoder, self).__init__()
        self._target_transform = NPBlockRelu2d(x_dim, hidden_dim, dropout)
        hidden_dim_2 = 2 * hidden_dim + latent_dim
        self._decoder = nn.Sequential(
            *[
                NPBlockRelu2d(hidden_dim_2, hidden_dim_2, dropout)
                for _ in range(n_decoder_layers)
            ]
        )
        self._mean = nn.Linear(2 * hidden_dim + latent_dim, y_dim)
        self._std = nn.Linear(2 * hidden_dim + latent_dim, y_dim)
        self.min_std = min_std

    def forward(self, r, z, target_x):
        x = self._target_transform(target_x)

        representation = torch.cat([torch.cat([r, z], dim=-1), x], dim=-1)
        representation = self._decoder(representation)

        mean = self._mean(representation)
        log_sigma = self._std(representation)

        log_sigma = F.softplus(log_sigma) + math.log(self.min_std)
        sigma = torch.exp(log_sigma)

        dist = torch.distributions.Normal(mean, sigma)
        return dist, log_sigma
