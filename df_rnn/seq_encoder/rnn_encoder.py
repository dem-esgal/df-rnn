import torch
from torch import nn as nn

from df_rnn.seq_encoder.abs_seq_encoder import AbsSeqEncoder
from df_rnn.seq_encoder.utils import LastStepEncoder
from df_rnn.trx_encoder import PaddedBatch, TrxEncoder


class RnnEncoder(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()

        self.hidden_size = config['hidden_size']
        self.rnn_type = config['type']
        self.bidirectional = config['bidir']
        self.trainable_starter = config['trainable_starter']
        self.dropout = config['dropout']
        rnn_args = {
            "input_size": input_size,
            "hidden_size": self.hidden_size,
            "num_layers": 1,
            "batch_first": True,
            "dropout": self.dropout
        }

        # initialize RNN
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(**rnn_args)
            if self.bidirectional:
                self.rnn2 = nn.LSTM(**rnn_args)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(**rnn_args)
            if self.bidirectional:
                self.rnn2 = nn.GRU(**rnn_args)
        else:
            raise Exception(f'wrong rnn type "{self.rnn_type}"')

        # initialize starter position if needed
        if self.trainable_starter == 'static':
            num_dir = 2 if self.bidirectional else 1
            self.starter_h = nn.Parameter(torch.randn(num_dir, 1, self.hidden_size))

    def forward(self, x: PaddedBatch, h_0: torch.Tensor = None):
        """

        :param x:
        :param h_0: None or [1, B, H] float tensor
                    0.0 values in all components of hidden state of specific client means no-previous state and
                    use starter for this client
                    h_0 = None means no-previous state for all clients in batch
        :return:
        """
        shape = x.payload.size()
        assert shape[1] > 0, "Batch can'not have 0 transactions"

        # prepare initial state
        if self.trainable_starter == 'static':
            starter_h = self.starter_h.expand(-1, shape[0], -1).contiguous()
            if h_0 is None:
                h_0 = starter_h
            elif h_0 is not None and not self.training:
                h_0 = torch.where(
                    (h_0.squeeze(0).abs().sum(dim=1) == 0.0).unsqueeze(0).unsqueeze(2).expand(*starter_h.size()),
                    starter_h,
                    h_0,
                )
            else:
                raise NotImplementedError('Unsupported mode: cannot mix fixed X and learning Starter')

        # pass-through rnn
        if self.rnn_type == 'lstm':
            out, _ = self.rnn(x.payload)
            if self.bidirectional:
                out2, _ = self.rnn2(torch.flip(x.payload, dims=(1,)))
                out = torch.cat((out, out2), dim=2)

        elif self.rnn_type == 'gru':
            out, _ = self.rnn(x.payload, h_0)
            if self.bidirectional:
                out2, _ = self.rnn2(torch.flip(x.payload, dims=(1,)), h_0)
                out = torch.cat((out, out2), dim=2)
        else:
            raise Exception(f'wrong rnn type "{self.rnn_type}"')

        return PaddedBatch(out, x.seq_lens)


class RnnSeqEncoder(AbsSeqEncoder):
    def __init__(self, params, is_reduce_sequence):
        super().__init__(params, is_reduce_sequence)

        p = TrxEncoder(params['trx_encoder'])
        e = RnnEncoder(p.output_size, params['rnn'])
        layers = [p, e]
        self.reducer = LastStepEncoder()
        self.model = torch.nn.Sequential(*layers)

    @property
    def category_max_size(self):
        return self.model[0].category_max_size

    @property
    def category_names(self):
        return self.model[0].category_names

    @property
    def embedding_size(self):
        return self.params['rnn']['hidden_size']

    def forward(self, x):
        x = self.model(x)
        if self.is_reduce_sequence:
            x = self.reducer(x)
        return x
