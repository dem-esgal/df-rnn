import torch

from torch.nn import Linear, BatchNorm1d, Sigmoid, Sequential, LogSoftmax, GELU, Parameter
from typing import List

from df_rnn.seq_encoder.rnn_encoder import RnnEncoder
from df_rnn.seq_encoder.utils import TimeStepShuffle, scoring_head
from df_rnn.trx_encoder import TrxEncoder
from df_rnn.custom_layers import Squeeze
from df_rnn.seq_encoder.utils import NormEncoder


def rnn_model(params):
    p = TrxEncoder(params['trx_encoder'])
    e = RnnEncoder(p.output_size, params['rnn'])
    h = scoring_head(
        input_size=params['rnn.hidden_size'] * (2 if params['rnn.bidir'] else 1),
        params=params['head']
    )

    m = torch.nn.Sequential(p, e, h)
    return m


def rnn_shuffle_model(params):
    p = TrxEncoder(params['trx_encoder'])
    p_size = p.output_size
    p = torch.nn.Sequential(p, TimeStepShuffle())
    e = RnnEncoder(p_size, params['rnn'])
    h = scoring_head(
        input_size=params['rnn.hidden_size'] * (2 if params['rnn.bidir'] else 1),
        params=params['head']
    )

    m = torch.nn.Sequential(p, e, h)
    return m


def freeze_layers(model):
    for p in model.parameters():
        p.requires_grad = False


def create_head_layers(params, seq_encoder=None):
    layers = []
    _locals = locals()
    for l_name, l_params in params['head_layers']:
        l_params = {k: int(v.format(**_locals)) if type(v) is str else v
                    for k, v in l_params.items()}

        cls = _locals.get(l_name, None)
        layers.append(cls(**l_params))
    return torch.nn.Sequential(*layers)


class Head(torch.nn.Module):
    r"""Head for the sequence encoder

    Parameters
    ----------
         input_size: int
            input size
         use_norm_encoder: bool. Default: False
            whether to use normalization layers before the head
         use_batch_norm: bool. Default: False.
            whether to use BatchNorm.
         hidden_layers_sizes: List[int]. Default: None.
            sizes of linear layers. If None without additional linear layers. Default = None,
         objective: str. Default: None.
            Options: None, 'classification', 'regression'. Default = None.
         num_classes: int. Default: 1.
            The number of classed in classification problem. Default correspond to binary classification.

     """

    def __init__(self,
                 input_size: int,
                 use_norm_encoder: bool = False,
                 use_batch_norm: bool = False,
                 hidden_layers_sizes: List[int] = None,
                 objective: str = None,
                 num_classes: int = 1,
                 rnn_bidirectional=False):
        super().__init__()
        layers = []
        self.input_size = input_size
        self.bidirectional = rnn_bidirectional

        if self.bidirectional:
            self.fc = Linear(input_size, input_size, False)
            self.W = Parameter(torch.ones(1, ), requires_grad=True)

        if hidden_layers_sizes is not None:
            layers_size = [input_size] + list(hidden_layers_sizes)
            for size_in, size_out in zip(layers_size[:-1], layers_size[1:]):
                layers.append(GELU())
                layers.append(Linear(size_in, size_out))
                if use_batch_norm:
                    layers.append(BatchNorm1d(size_out))

        if use_batch_norm:
            layers.append(BatchNorm1d(input_size))

        if use_norm_encoder:
            layers.append(NormEncoder())

        if objective == 'classification':
            if num_classes == 1:
                h = Sequential(Linear(input_size, num_classes), Sigmoid(), Squeeze())
            else:
                h = Sequential(Linear(input_size, num_classes), LogSoftmax(dim=1))
            layers.append(h)

        elif objective == 'regression':
            h = Sequential(Linear(input_size, 1), Squeeze())
            layers.append(h)

        elif objective is not None:
            raise AttributeError(f"Unknown objective {objective}. Supported: classification and regression")

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        if self.bidirectional:
            parts = x.split((self.input_size, self.input_size), 1)
            x = parts[0] * self.W + self.fc(parts[1]) * (1 - self.W)
        return self.model(x)
