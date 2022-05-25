import torch
from typing import List, Dict, Union
from df_rnn.seq_encoder.rnn_encoder import RnnSeqEncoder


def create_encoder(params, is_reduce_sequence=True):
    encoder_type = params['encoder_type']
    if encoder_type == 'rnn':
        return RnnSeqEncoder(params, is_reduce_sequence)

    raise AttributeError(f'Unknown encoder_type: "{encoder_type}"')


class SequenceEncoder(torch.nn.Module):
    r"""Sequence encoder

    Parameters
    ----------
     category_features: Dict[str, int],
     numeric_features: List[str],
     trx_embedding_size: int. Default: 16
        The number of features in the category fields embedding.
     trx_embedding_noise: float. Default: 0.0
        When > 0 applies additive Gaussian noise to category fields embeddings.
     encoder_type: str. Default: 'rnn'
       Type of encoder. Options: 'rnn' for RNN, 'transformer' for transformer, and 'agg_features' for
       the net that aggregated statistic from the sequence
     rnn_hidden_size: int. Default: 512
        The number of features in the RNN hidden state. Used only if encoder_type == 'rnn'.
     rnn_type: str. Default: 'gru'
        Type of RNN. Options: 'gru', 'lstm'. Used only if encoder_type == 'rnn'.
     rnn_trainable_starter: bool. Default: False
        Whether to use trainable starter for the RNN. Used only if encoder_type == 'rnn'.
     rnn_bidirectional: bool. Default: False
         If True, becomes a bidirectional RNN. Used only if encoder_type == 'rnn'.
     transformer_input_size: int. Default: 512
        Transformer input size, used only if encoder_type == 'transformer'.
     transformer_dim_hidden: int. Default: 256
        The number of features in the attention hidden state. Used only if encoder_type == 'transformer'.
     transformer_n_layers: int. Default: 4
        The number of layers in transformer. Used only if encoder_type == 'transformer'.
     transformer_n_heads: int. Default: 4
        The number of heads in each transformer attention layer. Used only if encoder_type == 'transformer'.
     transformer_shared_layers: bool. Default: False
        Whether to share weights in transformer layers on not. Used only if encoder_type == 'transformer'.
     transformer_use_after_mask: bool. Default: False
         Whether to share weights in transformer layers on not. Used only if encoder_type == 'transformer'.
     transformer_use_src_key_padding_mask: bool. Default: False
         Whether to use padding mask on not. Used only if encoder_type == 'transformer'.
     transformer_use_positional_encoding: bool. Default: False
         Whether to use positional encoding on not. Used only if encoder_type == 'transformer'.
     transformer_train_starter: bool. Default: False
        Whether to use trainable starter for the transformer. Used only if encoder_type == 'transformer'.
     transformer_dropout: float. Default: 0.0
        The dropout after attention layer value. Used only if encoder_type == 'transformer'.

     """

    def __init__(self,
                 category_features: Dict[str, int],
                 numeric_features: List[str],
                 trx_embedding_size: Union[int, Dict[str, int]] = 16,
                 trx_embedding_noise: float = 0.0,
                 encoder_type: str = 'rnn',
                 rnn_hidden_size: int = 512,
                 rnn_type: str = 'gru',
                 rnn_trainable_starter: bool = False,
                 rnn_bidirectional: bool = False,
                 rnn_dropout: float = 0.2):
        super().__init__()
        if isinstance(trx_embedding_size, dict):
            trx_embeddings = {k: {'in': v, 'out': trx_embedding_size[k]} for k, v in category_features.items()}
        else:
            trx_embeddings = {k: {'in': v, 'out': trx_embedding_size if trx_embedding_size > 0 else v} for k, v in
                              category_features.items()}

        trx_encoder_params = {
            'embeddings_noise': trx_embedding_noise,
            'embeddings': trx_embeddings,
            'numeric_values': {k: 'identity' for k in numeric_features},
        }

        params = {'trx_encoder': trx_encoder_params}

        if encoder_type == 'rnn':
            params['rnn'] = {
                'type': rnn_type,
                'hidden_size': rnn_hidden_size,
                'bidir': rnn_bidirectional,
                'trainable_starter': 'statis' if rnn_trainable_starter else None,
                'dropout': rnn_dropout
            }
            model = RnnSeqEncoder(params, True)
        else:
            raise AttributeError(f'Unknown encoder_type: {encoder_type}')

        self.model = model
        self.params = params

    @property
    def is_reduce_sequence(self):
        return self.model._is_reduce_sequence

    @is_reduce_sequence.setter
    def is_reduce_sequence(self, value):
        self.model._is_reduce_sequence = value

    @property
    def category_max_size(self):
        return self.model.category_max_size

    @property
    def category_names(self):
        return self.model.category_names

    @property
    def embedding_size(self):
        return self.model.embedding_size

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.model(x)
