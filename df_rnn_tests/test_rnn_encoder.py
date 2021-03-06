import torch

from df_rnn.seq_encoder.rnn_encoder import RnnEncoder
from df_rnn.trx_encoder import PaddedBatch
from df_rnn.seq_encoder.utils import LastStepEncoder


def get_data():
    return PaddedBatch(
        payload=torch.arange(4*5*8).view(4, 8, 5).float(),
        length=torch.tensor([4, 2, 6, 8])
    )


def test_shape():
    params = {
        'hidden_size': 6,
        'type': 'gru',
        'bidir': False,
        'trainable_starter': 'static',
        'dropout': 0.2
    }
    model = RnnEncoder(5, params)

    x = get_data()

    out = model(x)
    print(out.payload.shape)


def test_last_step():
    params = {
        'hidden_size': 6,
        'type': 'gru',
        'bidir': False,
        'trainable_starter': 'static',
        'dropout': 0.2
    }
    model = torch.nn.Sequential(RnnEncoder(5, params), LastStepEncoder())

    x = get_data()

    h = model(x)
    print(h.shape)
