import logging
from collections import defaultdict
import numpy as np
import torch
from df_rnn.data_load.iterable_processing_dataset import IterableProcessingDataset
from df_rnn.trx_encoder import PaddedBatch

logger = logging.getLogger(__name__)


def padded_collate(batch):
    new_x_ = defaultdict(list)
    for x, _ in batch:
        for k, v in x.items():
            new_x_[k].append(v)

    lengths = torch.IntTensor([len(e) for e in next(iter(new_x_.values()))])

    new_x = {k: torch.nn.utils.rnn.pad_sequence(v, batch_first=True) for k, v in new_x_.items()}
    new_y = np.array([y for _, y in batch])
    if new_y.dtype.kind in ('i', 'f'):
        new_y = torch.from_numpy(new_y)

    return PaddedBatch(new_x, lengths), new_y


def padded_collate_wo_target(batch):
    new_x_ = defaultdict(list)
    for x in batch:
        for k, v in x.items():
            new_x_[k].append(v)

    lengths = torch.IntTensor([len(e) for e in next(iter(new_x_.values()))])
    new_x = {k: torch.nn.utils.rnn.pad_sequence(v, batch_first=True) for k, v in new_x_.items()}
    return PaddedBatch(new_x, lengths)


class IterableChain:
    def __init__(self, *i_filters):
        self.i_filters = i_filters

    def __call__(self, seq):
        for f in self.i_filters:
            logger.debug(f'Applied {f} to {seq}')
            seq = f(seq)
        logger.debug(f'Returned {seq}')
        return seq
