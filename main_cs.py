import gc
import os
import pandas as pd
import torch
import pytorch_lightning as pl
import numpy as np

from df_rnn.data_preprocessing import PandasDataPreprocessor
from df_rnn.seq_encoder import SequenceEncoder
from df_rnn.models import Head
from df_rnn.lightning_modules.emb_module import EmbModule
from df_rnn.data_load.data_module.emb_data_module import train_data_loader, inference_data_loader, \
    EmbeddingTrainDataModule


def rlencode(x: np.array) -> np.array:
    n = len(x)
    if n == 0:
        return (np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([], dtype=x.dtype))

    starts = np.r_[0, np.flatnonzero(np.not_equal(x[1:], x[:-1])) + 1]
    lengths = np.diff(np.r_[starts, n])
    result = np.zeros_like(x)
    result[starts] = lengths

    return result


if __name__ == '__main__':
    data_path = '../data/'
    SAVE_LABEL_ENCODER = False

    clickstream = pd.read_csv(os.path.join(data_path, 'clickstream.csv'))
    clickstream.sort_values(by=['user_id', 'new_uid', 'timestamp'], inplace=True)
    clickstream.reset_index(inplace=True)
    print('sorted')
    uid_array = clickstream['new_uid'].to_numpy()
    clickstream['hash'] = pd.util.hash_array(uid_array).astype('int') + clickstream['cat_id']
    hash_array = clickstream['hash'].to_numpy()
    print('hashed')
    counts = rlencode(hash_array)
    print('encoded')
    clickstream['counts'] = counts
    clickstream = clickstream.iloc[counts != 0]
    print('saved')
    source_data = clickstream[['user_id', 'cat_id', 'timestamp', 'counts', 'new_uid']]

    del clickstream
    gc.collect()

    print(source_data.head())
    source_data['transaction_dttm'] = pd.to_datetime(source_data['timestamp'], format='%Y-%m-%d %H:%M:%S')
    source_data['dow'] = source_data['transaction_dttm'].dt.dayofweek
    source_data['transaction_dttm'] = (source_data['transaction_dttm'] - source_data[
        'transaction_dttm'].min()) / np.timedelta64(1, 's')

    print(source_data.shape)

    preprocessor = PandasDataPreprocessor(
        col_id='user_id',
        cols_event_time='transaction_dttm',
        time_transformation='float',
        cols_category=['cat_id', 'dow', 'new_uid'],
        cols_log_norm=[],
        cols_identity=['counts'],
        print_dataset_info=False,
    )

    train = preprocessor.fit_transform(source_data)
    if SAVE_LABEL_ENCODER:
        preprocessor.save('panda_cs_preprocessor.pkl')

    gc.collect()

    for item in train:
        item['uids'] = np.repeat(float(np.unique(item['new_uid']).shape[0]), item['new_uid'].shape[0])
        del item['new_uid']

    print(len(train))
    del preprocessor.cols_category_mapping['new_uid']

    rnn_bidirectional = False
    hidden_layers_sizes = None
    rnn_dropout = 0.15
    trx_embedding_noise = 0.003
    split_count = 2
    category_features = preprocessor.get_category_sizes()
    seq_encoder = SequenceEncoder(
        trx_embedding_size={'cat_id': 196, 'dow': 7},
        encoder_type='rnn',
        rnn_trainable_starter=False,
        rnn_bidirectional=rnn_bidirectional,
        rnn_dropout=rnn_dropout,
        category_features=preprocessor.get_category_sizes(),
        numeric_features=['uids', 'counts'],
        trx_embedding_noise=trx_embedding_noise
    )
    input_size = seq_encoder.embedding_size

    head = Head(input_size=input_size, use_norm_encoder=True, hidden_layers_sizes=hidden_layers_sizes,
                rnn_bidirectional=rnn_bidirectional)

    model = EmbModule(seq_encoder=seq_encoder, head=head,
                      k_in_top_k=split_count - 1,
                      lr=3e-3,
                      lr_scheduler_step_size=50,
                      lr_scheduler_step_gamma=0.8)

    ###################################################
    # Data module
    ###################################################

    dm = EmbeddingTrainDataModule(
        dataset=train,
        pl_module=model,
        min_seq_len=25,
        seq_split_strategy='SampleSlices',
        category_names=model.seq_encoder.category_names,
        category_max_size=model.seq_encoder.category_max_size,
        split_count=split_count,
        split_cnt_min=120,
        split_cnt_max=1000,
        train_num_workers=16,
        train_batch_size=300,
        valid_num_workers=16,
        valid_batch_size=300
    )

    trainer = pl.Trainer(
        default_root_dir='./logs_cs',
        accumulate_grad_batches=16,
        num_sanity_val_steps=0,
        max_epochs=1200,
        gpus=1 if torch.cuda.is_available() else 0,
    )

    trainer.fit(model, dm)
