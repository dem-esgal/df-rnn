import os
import pandas as pd
import torch
import pytorch_lightning as pl
import numpy as np

from df_rnn.data_load.data_module.base_data_module import BaseDataModule
from df_rnn.data_preprocessing import PandasDataPreprocessor
from df_rnn.lightning_modules.cls_module import ClsModule
from df_rnn.seq_encoder import SequenceEncoder
from df_rnn.models import Head

if __name__ == '__main__':
    checkpoint = torch.load('./lightning_logs/version_176/checkpoints/epoch=999-step=14999.ckpt')
    checkpoint['state_dict']['_head.model.1.0.weight'] = torch.rand(1, 512)
    checkpoint['state_dict']['_head.model.1.0.bias'] = torch.zeros((1,))

    LABEL = 'higher_education'
    data_path = '../data/'
    train_target = pd.read_csv(os.path.join(data_path, 'train.csv'))
    train_target[LABEL] = train_target[LABEL].astype('float16')
    train_target = train_target.set_index('bank').to_dict()[LABEL]

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

    dataset = preprocessor.fit_transform(source_data)
    train = []
    test = []

    for item in dataset:
        if item['user_id'] in train_target:
            item[LABEL] = train_target[item['user_id']]
            train.append(item)
        else:
            test.append(item)

    print(len(train), len(test))
    rnn_bidirectional = True
    hidden_layers_sizes = None
    rnn_dropout = 0.1
    trx_embedding_noise = 0.003
    category_features = preprocessor.get_category_sizes()

    seq_encoder = SequenceEncoder(
        trx_embedding_size={'mcc_code': 128, 'currency_rk': 4, 'dow': 7},
        encoder_type='rnn',
        rnn_trainable_starter=False,
        rnn_bidirectional=rnn_bidirectional,
        rnn_dropout=rnn_dropout,
        category_features=preprocessor.get_category_sizes(),
        numeric_features=['income', 'outcome'],
        trx_embedding_noise=trx_embedding_noise)

    input_size = seq_encoder.embedding_size

    head = Head(
        input_size=input_size,
        num_classes=1,
        objective='classification',
        use_norm_encoder=True,
        hidden_layers_sizes=hidden_layers_sizes,
        rnn_bidirectional=rnn_bidirectional)

    model = ClsModule(seq_encoder=seq_encoder,
                      head=head,
                      lr=1e-4,
                      lr_scheduler_step_size=5,
                      lr_scheduler_step_gamma=0.5)

    ###################################################
    # Data module
    ###################################################

    dm = BaseDataModule(
        dataset=train,
        dataset_test=test,
        pl_module=model,
        min_seq_len=25,
        train_num_workers=16,
        train_batch_size=128,
        valid_num_workers=16,
        valid_batch_size=128,
        target_col=LABEL,
    )

    model.load_state_dict(checkpoint['state_dict'])

    trainer = pl.Trainer(
        default_root_dir='./education_task',
        num_sanity_val_steps=0,
        max_epochs=40,
        gpus=1 if torch.cuda.is_available() else 0,
    )

    trainer.fit(model, dm)

    result = trainer.predict(model, dm)
    cntr = 0
    for batch in result:
        batch_numpy = batch.detach().cpu().numpy()
        for i in range(0, batch_numpy.shape[0]):
            test[cntr][LABEL] = batch_numpy[i]
            cntr += 1


    df = pd.DataFrame.from_records(test)
    print(df.head())
    df['higher_education_proba'] = df[LABEL]
    df['bank'] = df['user_id']
    df[['bank', 'higher_education_proba']].to_csv('sub1.csv', sep=',', index=False)

