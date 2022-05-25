import torch
from torch import nn as nn

from df_rnn.custom_layers import Squeeze
from df_rnn.trx_encoder import PaddedBatch

"""
Different simple torch modules.
"""
class ConcatLenEncoder(nn.Module):
    def forward(self, x: PaddedBatch):
        lens = x.seq_lens.unsqueeze(-1).float().to(x.payload.device)
        lens_normed = lens / 200

        h = x.payload[range(len(x.payload)), [l - 1 for l in x.seq_lens]]

        embeddings = torch.cat([h, lens_normed, -torch.log(lens_normed)], -1)
        return embeddings


class TimeStepShuffle(nn.Module):
    def forward(self, x: PaddedBatch):
        shuffled = []
        for seq, slen in zip(x.payload, x.seq_lens):
            idx = torch.randperm(slen) + 1
            pad_idx = torch.arange(slen + 1, len(seq))
            idx = torch.cat([torch.zeros(1, dtype=torch.long), idx, pad_idx])
            shuffled.append(seq[idx])

        shuffled = PaddedBatch(torch.stack(shuffled), x.seq_lens)
        return shuffled


class LastStepEncoder(nn.Module):
    def forward(self, x: PaddedBatch):
        h = x.payload[range(len(x.payload)), [l - 1 for l in x.seq_lens]]
        return h


class NormEncoder(nn.Module):
    def __init__(self, eps=1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return x / (x.pow(2).sum(dim=1) + self.eps).pow(0.5).unsqueeze(-1).expand(*x.size())


class AllStepsHead(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, x: PaddedBatch):
        out = self.head(x.payload)
        return PaddedBatch(out, x.seq_lens)


class AllStepsMeanHead(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, x: PaddedBatch):
        out = self.head(x.payload)
        means = torch.tensor([e[0:l].mean() for e, l in zip(out, x.seq_lens)])
        return means


class FlattenHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: PaddedBatch):
        mask = torch.zeros(x.payload.shape, dtype=bool)
        for i, length in enumerate(x.seq_lens):
            mask[i, :length] = True

        return x.payload.flatten()[mask.flatten()]


def scoring_head(input_size, params):
    if params['pred_all_states'] and params['explicit_lengths']:
        raise AttributeError('Can not use `pred_all_states` and `explicit_lengths` together for now')

    layers = []

    if not params['pred_all_states']:
        if params['explicit_lengths']:
            e0 = ConcatLenEncoder()
            input_size += 2
            layers.append(e0)
        else:
            e0 = LastStepEncoder()
            layers.append(e0)

    if params['norm_input']:
        layers.append(NormEncoder())

    if params['use_batch_norm']:
        layers.append(nn.BatchNorm1d(input_size))

    if "head_layers" not in params:
        head_output_size = params.get('num_classes', 1)
        if params.get('objective', 'classification') == 'classification':
            if head_output_size == 1:
                h = nn.Sequential(nn.Linear(input_size, head_output_size), nn.Sigmoid(), Squeeze())
            else:
                h = nn.Sequential(nn.Linear(input_size, head_output_size), nn.LogSoftmax(dim=1))
        else:
            h = nn.Sequential(nn.Linear(input_size, head_output_size), Squeeze())
        if params['pred_all_states']:
            if params['pred_all_states_mean']:
                h = AllStepsMeanHead(h)
                layers.append(h)
            else:
                h = AllStepsHead(h)
                layers.append(h)
                if params.get('pred_flatten', False):
                    layers.append(FlattenHead())

        else:
            layers.append(h)
    else:
        head_layers = [input_size] + params["head_layers"] + [1]
        for size_in, size_out in zip(head_layers[:-1], head_layers[1:]):
            layers.append(nn.Linear(size_in, size_out))
            if size_out != 1:
                layers.append(nn.BatchNorm1d(size_out))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
            else:
                layers.append(nn.Sigmoid())
        layers.append(Squeeze())

    return nn.Sequential(*layers)
