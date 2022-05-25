import os
import pandas as pd
import torch
import pytorch_lightning as pl
import numpy as np

from df_rnn.data_preprocessing import PandasDataPreprocessor
from df_rnn.seq_encoder import SequenceEncoder
from df_rnn.models import Head
from df_rnn.lightning_modules.emb_module import EmbModule
from df_rnn.data_load.data_module.emb_data_module import EmbeddingTrainDataModule

if __name__ == '__main__':
    data_path = '../data/'
    SAVE_LABEL_ENCODER = False

    source_data = pd.read_csv(os.path.join(data_path, 'transactions.csv'))
    source_data['transaction_dttm'] = pd.to_datetime(source_data['transaction_dttm'], format='%Y-%m-%d %H:%M:%S')
    source_data['dow'] = source_data['transaction_dttm'].dt.dayofweek

    source_data['transaction_dttm'] = (source_data['transaction_dttm'] - source_data[
        'transaction_dttm'].min()) / np.timedelta64(1, 's')
    source_data['income'] = source_data['transaction_amt'].apply(lambda x: x if x > 0 else 0)
    source_data['outcome'] = source_data['transaction_amt'].apply(lambda x: -x if x < 0 else 0)
    source_data.drop(['transaction_amt'], inplace=True, axis=1)
    print(source_data.shape)

    preprocessor = PandasDataPreprocessor(
        col_id='user_id',
        cols_event_time='transaction_dttm',
        time_transformation='float',
        cols_category=['mcc_code', 'currency_rk', 'dow'],
        cols_log_norm=['income', 'outcome'],
        cols_identity=[],
        print_dataset_info=False,
    )

    train = preprocessor.fit_transform(source_data)
    if SAVE_LABEL_ENCODER:
        preprocessor.save('panda_data_preprocessor.pkl')

    rnn_bidirectional = True
    hidden_layers_sizes = None
    rnn_dropout = 0.1
    trx_embedding_noise = 0.003
    split_count = 2
    category_features = preprocessor.get_category_sizes()

    seq_encoder = SequenceEncoder(
        trx_embedding_size={'mcc_code': 128, 'currency_rk': 4, 'dow': 7},
        encoder_type='rnn',
        rnn_trainable_starter=False,
        rnn_bidirectional=rnn_bidirectional,
        rnn_dropout=rnn_dropout,
        category_features=preprocessor.get_category_sizes(),
        numeric_features=['income', 'outcome'],
        trx_embedding_noise=trx_embedding_noise
    )
    input_size = seq_encoder.embedding_size

    head = Head(input_size=input_size, use_norm_encoder=True, hidden_layers_sizes=hidden_layers_sizes,
                rnn_bidirectional=rnn_bidirectional)

    model = EmbModule(seq_encoder=seq_encoder, head=head,
                      k_in_top_k=split_count - 1,
                      lr=3e-3,
                      lr_scheduler_step_size=200,
                      lr_scheduler_step_gamma=0.5)

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
        split_cnt_min=64,
        split_cnt_max=920,
        train_num_workers=16,
        train_batch_size=240,
        valid_num_workers=16,
        valid_batch_size=240
    )

    trainer = pl.Trainer(
        accumulate_grad_batches=6,
        num_sanity_val_steps=0,
        max_epochs=1000,
        gpus=1 if torch.cuda.is_available() else 0,
    )

    trainer.fit(model, dm)
