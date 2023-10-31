# %%
import pandas as pd
from IPython.display import display
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVR
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from Utils.model_utils import score
import numpy as np
import os

train_path = '/home/yang/Desktop/workspace/Project/AICUP_2023_ESUN_CreditCard/TrainingData_group/training_data.pkl'
with open(train_path, 'rb') as f:
    df = pickle.load(f)

# test_path = '/home/yang/Desktop/workspace/Project/AICUP_2023_ESUN_CreditCard/TrainData_simulation_distributed_group/general_test_data.pkl'
# with open(test_path, 'rb') as f:
#     test_df = pickle.load(f)


d = df[df["label"]==1].copy()
b = resample(df[df["label"]==0], n_samples=d.shape[0]*200, random_state=1)
ori_sub_df = pd.concat([d, b])
# ori_sub_df = resample(df, n_samples=int(df.shape[0]*0.1), random_state=1)
# ori_sub_test_df = resample(test_df, n_samples=int(test_df.shape[0]*0.1))

# %%
# process_df = pd.concat([ori_sub_df, ori_sub_test_df]).reset_index(drop=True)
process_df = ori_sub_df.copy().reset_index(drop=True)
# %%
# data preprocess
# 去除數量稀少的資料
# target = ['stocn']
# # print('null count: ' + df[df[target].isna()].shape[0])
# for col in target:
#     target_df = df.groupby(col).count().iloc[:, [0]]
#     target_df = target_df.rename(columns={'txkey':'0'})
#     display(target_df)

# for col in target:
#     target_label_df = df.groupby([col, 'label']).count().iloc[:, [0]]
#     target_label_df = target_label_df.rename(columns={'txkey':'0'})
#     display(target_label_df)

# for col in target:
#     target_label_df_t = df.groupby(['label', col]).count().iloc[:, [0]]
#     target_label_df_t = target_label_df_t.rename(columns={'txkey':'0'})
#     display(target_label_df_t)

# t1 = []
# t0 = []
# for a, b in target_label_df.index:
#     if b == 1:
#         t1.append((a,b))
#     else:
#         t0.append((a,b))

# %%
# loctm 計算時間並轉換成類別
def time_col_complement(x):
    x = str(x)
    while len(x) < 6:
        x = '0'+x
    return x

def time_to_label(x):
    label_dict = {
        1: ['000000', '040000'],
        2: ['040000', '080000'],
        3: ['080000', '120000'],
        4: ['120000', '160000'],
        5: ['160000', '200000'],
        6: ['200000', '240000'],
    }
    for k, v in label_dict.items():
        if x >= v[0] and x < v[1]:
            return int(k)

process_df['loctm'] = process_df['loctm'].apply(lambda x: time_col_complement(x))
process_df['loctm'] = process_df['loctm'].apply(lambda x: time_to_label(x))

# %%
# mchno 如果是僅出現盜刷的特店代號就留下，其餘轉0
# target_label_df_t = df.groupby(['label', 'mchno']).count().iloc[:, [0]]
# target_label_df_t = target_label_df_t.rename(columns={'txkey':'0'})

# t1 = []
# t0 = []
# for a, b in target_label_df_t.index:
#     if b == 1:
#         t1.append((a,b))
#     else:
#         t0.append((a,b))
# set_t1 = set([k for k, v in t1])
# set_t0 = set([k for k, v in t0])
# only_1_mchno = list(set_t1 - set_t0)

# process_df['mchno'] = process_df['mchno'].apply(lambda x: 'SPECIAL' if x in only_1_mchno else 'NA')

# %%
# csmam 當消費地金額跟conam對不齊時，特別標注(非台灣幣值)
# 效果不好
process_df['csmam'] = process_df.apply(lambda x: x['csmam'] if x['csmam']!=x['conam'] else -1, axis=1)

# %%
# stscd 狀態碼為1的留著，其餘更改為0
# process_df["stscd"] = process_df["stscd"].apply(lambda x: x if x == 1 else 0)

# %%
# chid encode為刷卡次數
chid_dict = process_df.groupby('chid').count()['txkey'].to_dict()
process_df["chid_new"] = process_df["chid"].apply(lambda x: chid_dict[x] if x in chid_dict.keys() else 0)

with open('chid_dict.pkl', 'wb') as f:
    pickle.dump(chid_dict, f)

# %%
# chid_fraud 新增欄位為被盜刷次數
chid_fraud_dict = df[df['label']==1].groupby('chid').count()["txkey"].to_dict()
process_df["chid_fraud"] = process_df["chid"].apply(lambda x: chid_fraud_dict[x] if x in chid_fraud_dict.keys() else 0)

with open('chid_fraud_dict.pkl', 'wb') as f:
    pickle.dump(chid_fraud_dict, f)

# %%
# etymd 交易型態 用眾數補Null
etymd_mode = df['etymd'].mode()[0]
with open('etymd_mode.csv', 'w') as f:
    f.write(str(etymd_mode))
process_df['etymd'] = process_df['etymd'].fillna(etymd_mode)

# mcc 直接補0
process_df['mcc'] = process_df['mcc'].fillna(0)

# hcefg 支付型態 用眾數補Null
hcefg_mode = df['hcefg'].mode()[0]
with open('hcefg_mode.csv', 'w') as f:
    f.write(str(hcefg_mode))
process_df['hcefg'] = process_df['hcefg'].fillna(hcefg_mode)

# 用LinearRegression模型補 csmcu Null
with open('csmcu_null_model.pkl', 'rb') as f:
    csmcu_null_model = pickle.load(f)

pred_col = ['stocn', 'conam']
null_idx = np.where(process_df["csmcu"].isna())[0]
null_df = process_df.loc[null_idx]
x = null_df[pred_col].values
csmcu_pred_list = csmcu_null_model.predict(x)
process_df.loc[null_idx, 'csmcu'] = csmcu_pred_list

# 組合stocn 消費地國別 scity 消費城市欄位
process_df['stocn'] = process_df['stocn'].astype(str)
process_df["scity"] = process_df["scity"].astype(str)
process_df['new_location'] = process_df['stocn'] + process_df["scity"]

# %%
# train_col = ['txkey', 'locdt', 'loctm', 'chid', 'cano', 'contp', 'etymd', 'mchno',
#        'acqic', 'mcc', 'conam', 'ecfg', 'insfg', 'iterm', 'bnsfg', 'flam1',
#        'stocn', 'scity', 'stscd', 'ovrlt', 'flbmk', 'hcefg', 'csmcu', 'csmam',
#        'flg_3dsmk']
train_col = ['locdt', 'loctm', 'contp', 'etymd', 'mchno',
       'acqic', 'mcc', 'conam', 'ecfg', 'insfg', 'iterm', 'bnsfg', 'flam1',
       'stscd', 'ovrlt', 'flbmk', 'hcefg', 'csmcu', 'csmam',
       'flg_3dsmk', 'chid', 'new_location','chid_fraud', 'chid_new']
# train_col = ['locdt', 'loctm', 'contp', 'etymd', 'mchno',
#        'acqic', 'mcc', 'conam', 'ecfg', 'insfg', 'iterm', 'bnsfg', 'flam1',
#        'stocn', 'scity', 'stscd', 'ovrlt', 'flbmk', 'hcefg', 'csmcu', 'csmam',
#        'flg_3dsmk', 'chid_fraud']

process_df = process_df[train_col + ['label']].copy()

# %%
# label encode
# label_encode_col_list = ['txkey', 'chid', 'cano', 'contp', 'etymd', 'mchno', 'acqic', 'mcc', 'ecfg', 'insfg', 'bnsfg', 'stocn', 'scity', 'stscd', 'ovrlt', 'flbmk', 'hcefg', 'csmcu', 'flg_3dsmk']
label_encode_col_list = ['mchno', 'chid', 'contp', 'etymd', 'acqic', 'mcc', 'ecfg', 'insfg', 'bnsfg', 'stscd', 'ovrlt', 'flbmk', 'hcefg', 'csmcu', 'flg_3dsmk', 'new_location']
col_encoding = {}
for col in label_encode_col_list:
    labelencode = LabelEncoder()
    col_encoding[col] = labelencode.fit(process_df[col])
    process_df[col] = col_encoding[col].transform(process_df[col])

if os.path.exists('label_encode.pkl'):
    os.remove('label_encode.pkl')
with open('label_encode.pkl', 'wb') as f:
    pickle.dump(col_encoding, f)

#%%
# 用另外訓練的隨機森林補stscd值
# stscd_model_path = '/home/yang/Desktop/workspace/Project/AICUP_2023_ESUN_CreditCard/stscd_null_model.pkl'
# with open(stscd_model_path, 'rb') as f:
#     stscd_model = pickle.load(f)

# col_for_null_model = ['locdt', 'loctm', 'contp', 'etymd', 'mchno',
#        'acqic', 'mcc', 'conam', 'ecfg', 'insfg', 'iterm', 'bnsfg', 'flam1',
#        'stocn', 'scity', 'ovrlt', 'flbmk', 'hcefg', 'csmcu', 'csmam',
#        'flg_3dsmk']

# process_df['stscd'] = stscd_model.predict(process_df[col_for_null_model].values)
# %%
# one hot encode
# one_hot_encode_col_list = ['loctm']
# other_list = list(set(process_df.columns) - set(one_hot_encode_col_list))
# for col in other_list:
#     process_df[col] = process_df[col].astype(float)
# for col in one_hot_encode_col_list:
#     process_df[col] = process_df[col].astype(str)

# process_df = pd.get_dummies(process_df)

# %%
sub_df = process_df.loc[:ori_sub_df.shape[0]-1, :]
# sub_test_df = process_df.loc[ori_sub_df.shape[0]:, :]

# %%
sub_df_train = sub_df.copy()
sub_df_train_ans = sub_df_train['label']
del sub_df_train['label']
X_train, X_test, y_train, y_test = train_test_split(sub_df_train, sub_df_train_ans, test_size=0.2)

# %%
X_train_val = X_train.values
X_test_val = X_test.values
y_train_val = y_train.values.ravel()
y_test_val = y_test.values.ravel()

model = XGBClassifier()
model.fit(X_train_val, y_train_val)

pred = model.predict(X_train_val)
result = score(y_train_val, pred, f'[train]')

pred_test = model.predict(X_test_val)
result = score(y_test_val, pred_test, f'[val]')

# pred_col = train_col
# # pred_col = list(sub_test_df.columns)
# # pred_col.remove('label')
# data = sub_test_df[pred_col].values
# ans = sub_test_df['label'].values
# general_pred = model.predict(data)
# result = score(ans, general_pred, f'[val]')

if os.path.exists('model.pkl'):
    os.remove('model.pkl')
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
# %%
