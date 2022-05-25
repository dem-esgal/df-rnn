import random
from collections import defaultdict

import pytorch_lightning as pl
import torch

from df_rnn.data_load.iterable_processing.seq_len_filter import SeqLenFilter
from df_rnn.data_load.iterable_processing.feature_filter import FeatureFilter
from df_rnn.data_load.iterable_processing.category_size_clip import CategorySizeClip
from df_rnn.data_load.iterable_processing.target_move import TargetMove
from df_rnn.data_load.iterable_processing.to_torch_tensor import ToTorch
from df_rnn.data_load import IterableChain
from torch.utils.data import DataLoader
from df_rnn.data_load import padded_collate
from typing import List, Dict
from sklearn.model_selection import train_test_split

from df_rnn.trx_encoder import PaddedBatch


def padded_collate_train(batch):
    max_len = random.randint(100, 500)
    for item in batch:
        features = item[0] if type(item) is tuple else item
        len_ = 0
        rand_start = 0
        for k, v in features.items():
            if len_ == 0:
                len_ = v.shape[0]
                if len_ > max_len:
                    max_ = len_ - max_len
                    rand_start = random.randint(1, max_)

            if rand_start != 0:
                features[k] = v[rand_start: rand_start + max_len]

    return padded_collate(batch)


def padded_collate_predict(batch):
    new_x_ = defaultdict(list)
    for x in batch:
        for k, v in x.items():
            new_x_[k].append(v)

    lengths = torch.IntTensor([len(e) for e in next(iter(new_x_.values()))])
    new_x = {k: torch.nn.utils.rnn.pad_sequence(v, batch_first=True) for k, v in new_x_.items()}

    return PaddedBatch(new_x, lengths)


class BaseDataModule(pl.LightningDataModule):
    r"""pytorch-lightning data module for supervised training
    Parameters
    ----------
     dataset: List[Dict]
        dataset
     pl_module: pl.LightningModule
        Pytorch-lightning module used for training seq_encoder.
     min_seq_len: int. Default: 0.
        The minimal length of sequences used for training. The shorter sequences would be skipped.
     valid_size: float. Default: 0.05.
        Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
     train_num_workers: int. Default: 0.
        The number of workers for train dataloader. 0 = single-process loader
     train_batch_size: int. Default: 256.
        The number of samples (before splitting to subsequences) in each batch during training.
     valid_num_workers: int. Default: 0.
        The number of workers for validation dataloader. 0 = single-process loader       
     valid_batch_size: int. Default: 256.
        The number of samples (before splitting to subsequences) in each batch during validation.
     target_col: str. Default: 'target'.
        The name of target column.
     random_state : int. Default : 42.
        Controls the shuffling applied to the data before applying the split.
     """

    def __init__(self,
                 dataset: List[dict],
                 dataset_test: List[dict],
                 pl_module: pl.LightningModule,
                 min_seq_len: int = 0,
                 valid_size: float = 0.05,
                 train_num_workers: int = 0,
                 train_batch_size: int = 256,
                 valid_num_workers: int = 0,
                 valid_batch_size: int = 256,
                 target_col: str = 'target',
                 random_state: int = 42):

        super().__init__()
        self.dataset_test = dataset_test
        self.dataset_train, self.dataset_valid = train_test_split(dataset,
                                                                  test_size=valid_size,
                                                                  random_state=random_state)
        self.min_seq_len = min_seq_len
        self.train_num_workers = train_num_workers
        self.train_batch_size = train_batch_size
        self.valid_num_workers = valid_num_workers
        self.valid_batch_size = valid_batch_size
        self.keep_features = pl_module.seq_encoder.category_names
        self.keep_features.add('event_time')
        self.category_max_size = pl_module.seq_encoder.category_max_size
        self.target_col = target_col
        self.post_proc_train = IterableChain(*self.build_iterable_processing('train'))
        self.post_proc_valid = IterableChain(*self.build_iterable_processing('valid'))
        self.post_proc_predict = IterableChain(*self.build_iterable_processing('predict'))

        self.split_strategy_params = {
            'split_strategy': 'SampleSlices',
            'split_count': 2,
            'cnt_min': 1,
            'cnt_max': 2000
        }

    def prepare_data(self):
        self.dataset_train = list(self.post_proc_train(iter(self.dataset_train)))
        self.dataset_valid = list(self.post_proc_valid(iter(self.dataset_valid)))
        self.dataset_test = list(self.post_proc_predict(iter(self.dataset_test)))

    def build_iterable_processing(self, part):
        if part == 'train':
            yield SeqLenFilter(min_seq_len=self.min_seq_len)

        yield ToTorch()
        if part != 'predict':
            yield TargetMove(self.target_col)

        yield FeatureFilter(keep_feature_names=self.keep_features)
        yield CategorySizeClip(self.category_max_size)

    def train_dataloader(self):
        return DataLoader(
            shuffle=True,
            dataset=self.dataset_train,
            collate_fn=padded_collate_train,
            num_workers=self.train_num_workers,
            batch_size=self.train_batch_size
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset_valid,
            collate_fn=padded_collate,
            num_workers=self.valid_num_workers,
            batch_size=self.valid_batch_size
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.dataset_test,
            collate_fn=padded_collate_predict,
            num_workers=self.valid_num_workers,
            batch_size=self.valid_batch_size
        )
