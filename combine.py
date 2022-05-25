import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from catboost import Pool

transactions = pd.read_csv('embedings_t.csv')
transactions.rename(columns={transactions.columns[0]: 'bank'}, inplace=True)
clickstream = pd.read_csv('embedings_cs.csv')
clickstream.rename(columns={clickstream.columns[0]: 'rtk'}, inplace=True)

all_dicts = {}
all_dicts['rtk_le'] = LabelEncoder().fit(clickstream['rtk'])
clickstream['rtk'] = all_dicts['rtk_le'].transform(clickstream['rtk']) + 1
clickstream_dtypes = {'rtk': np.int16}
clickstream = clickstream.astype(clickstream_dtypes)

# Encodes user_id with numbers.
all_dicts['bank_le'] = LabelEncoder().fit(transactions['bank'])
transactions['bank'] = all_dicts['bank_le'].transform(transactions['bank']) + 1
transactions_dtypes = {'bank': np.int16}
transactions = transactions.astype(transactions_dtypes)

puzzle = pd.read_csv('../data/puzzle.csv')
print(puzzle.shape)
puzzle.head(2)

bank_dict = dict(zip(all_dicts['bank_le'].classes_, all_dicts['bank_le'].transform(all_dicts['bank_le'].classes_)))
puzzle['bank'] = puzzle['bank'].apply(lambda x: bank_dict.get(x, -1) + 1)
rtk_dict = dict(zip(all_dicts['rtk_le'].classes_, all_dicts['rtk_le'].transform(all_dicts['rtk_le'].classes_)))
puzzle['rtk'] = puzzle['rtk'].apply(lambda x: rtk_dict.get(x, -1) + 1)

train = pd.read_csv('../data/train_matching.csv')
print(train.shape)
train.head(2)

train['bank'] = all_dicts['bank_le'].transform(train['bank']) + 1
train.loc[train.rtk == '0', 'rtk'] = 0
train.loc[train.rtk != 0, 'rtk'] = train.loc[train.rtk != 0, 'rtk'].apply(lambda x: rtk_dict.get(x, -1) + 1)

# k invalid samples for a valid one.
k = 1
cor_dict = train.set_index('bank')['rtk'].to_dict()
NO_RTK = '0'
train_bank_ids = train[(train.rtk != NO_RTK)]['bank']
train_rtk_ids = train[train.bank.isin(train_bank_ids)]['rtk'].drop_duplicates()
df_train = pd.DataFrame(train_bank_ids, columns=['bank'])
df_train['rtk'] = df_train['bank'].apply(lambda x: cor_dict[x])
dfs = [df_train]
for i in range(k):
    df_train2 = df_train.copy()
    df_train2['rtk'] = df_train2['bank'].apply(lambda x: train_rtk_ids.sample(1).values.tolist()[0])
    dfs.append(df_train2)
df_train = pd.concat(dfs)
train['bank+rtk'] = train['bank'].astype('str') + '_' + train['rtk'].astype('str')
df_train['bank+rtk'] = df_train['bank'].astype('str') + '_' + df_train['rtk'].astype('str')
df_train['target'] = df_train['bank+rtk'].isin(train['bank+rtk']).astype('int')

df_train.drop_duplicates('bank+rtk', inplace=True)
df_train.reset_index(inplace=True, drop=True)

X_train = df_train.merge(transactions, how='left', left_on='bank', right_on='bank') \
    .merge(clickstream, how='left', left_on='rtk', right_on='rtk'
           ).fillna(0)
uniq = X_train['bank'].unique()
[train_df, eval_df] = train_test_split(uniq, test_size=0.05, random_state=123)
train_df = X_train[X_train['bank'].isin(train_df)]
eval_df = X_train[X_train['bank'].isin(eval_df)]

train_data = Pool(train_df.drop(['bank', 'rtk', 'bank+rtk', 'target'], axis=1), label=train_df['target'],
                  has_header=True)
eval_data = Pool(eval_df.drop(['bank', 'rtk', 'bank+rtk', 'target'], axis=1), label=eval_df['target'], has_header=True)

print(X_train.head())

clf = CatBoostClassifier(iterations=5000,
                         depth=7, learning_rate=0.01, loss_function='CrossEntropy', eval_metric='Accuracy')
clf.fit(train_data, eval_set=eval_data, metric_period=50, verbose=50, early_stopping_rounds=200)
probs = clf.predict_proba(eval_data)
name = 'matching_model_3'
clf.save_model(f'./container/{name}.cbm')
