# %%
import pandas as pd
from IPython.display import display
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from Utils.model_utils import score

train_path = '/home/yang/Desktop/workspace/Project/AICUP_2023_ESUN_CreditCard/TrainData_simulation_distributed_group/training_data.pkl'
with open(train_path, 'rb') as f:
    df = pickle.load(f)

test_path = '/home/yang/Desktop/workspace/Project/AICUP_2023_ESUN_CreditCard/TrainData_simulation_distributed_group/general_test_data.pkl'
with open(test_path, 'rb') as f:
    test_df = pickle.load(f)

ori_sub_df = resample(df, n_samples=int(df.shape[0]*0.05))
ori_sub_test_df = resample(test_df, n_samples=int(test_df.shape[0]*0.1))

# %%
process_df = pd.concat([ori_sub_df, ori_sub_test_df]).reset_index(drop=True)
# %%
# data preprocess
# 去除數量稀少的資料
target = ['csmam']
for col in target:
    target_df = df.groupby(col).count().iloc[:, [0]]
    target_df = target_df.rename(columns={'txkey':'0'})
    display(target_df)

for col in target:
    target_label_df = df.groupby([col, 'label']).count().iloc[:, [0]]
    target_label_df = target_label_df.rename(columns={'txkey':'0'})
    display(target_label_df)

for col in target:
    target_label_df_t = df.groupby(['label', col]).count().iloc[:, [0]]
    target_label_df_t = target_label_df_t.rename(columns={'txkey':'0'})
    display(target_label_df_t)

t1 = []
t0 = []
for a, b in target_label_df.index:
    if b == 1:
        t1.append((a,b))
    else:
        t0.append((a,b))

# %%
# loctm 計算時間並轉換成類別
def time_col_complement(x):
    x = str(x)
    while len(x) < 6:
        x = '0'+x
    return x

def time_to_label(x):
    label_dict = {
        '1': ['000000', '040000'],
        '2': ['040000', '080000'],
        '3': ['080000', '120000'],
        '4': ['120000', '160000'],
        '5': ['160000', '200000'],
        '6': ['200000', '240000'],
    }
    for k, v in label_dict.items():
        if x >= v[0] and x < v[1]:
            return int(k)

process_df['loctm'] = process_df['loctm'].apply(lambda x: time_col_complement(x))
process_df['loctm'] = process_df['loctm'].apply(lambda x: time_to_label(x))

# %%
# mchno 如果是僅出現盜刷的特店代號就留下，其餘轉0
target_label_df_t = df.groupby(['label', 'mchno']).count().iloc[:, [0]]
target_label_df_t = target_label_df_t.rename(columns={'txkey':'0'})

t1 = []
t0 = []
for a, b in target_label_df_t.index:
    if b == 1:
        t1.append((a,b))
    else:
        t0.append((a,b))
set_t1 = set([k for k, v in t1])
set_t0 = set([k for k, v in t0])
only_1_mchno = list(set_t1 - set_t0)

process_df['mchno'] = process_df['mchno'].apply(lambda x: 'SPECIAL' if x in only_1_mchno else 'NA')

# %%
# csmam 當消費地金額跟conam對不齊時，特別標注(非台灣幣值)
# 效果不好
process_df['new_location'] = process_df.apply(lambda x: 'SPECIAL' if x['csmam']!=x['conam'] else 'NA', axis=1)

# %%
# label encode
# label_encode_col_list = ['txkey', 'chid', 'cano', 'contp', 'etymd', 'mchno', 'acqic', 'mcc', 'ecfg', 'insfg', 'bnsfg', 'stocn', 'scity', 'stscd', 'ovrlt', 'flbmk', 'hcefg', 'csmcu', 'flg_3dsmk']
label_encode_col_list = ['new_location', 'contp', 'etymd', 'mchno', 'acqic', 'mcc', 'ecfg', 'insfg', 'bnsfg', 'stocn', 'scity', 'stscd', 'ovrlt', 'flbmk', 'hcefg', 'csmcu', 'flg_3dsmk']
labelencode = LabelEncoder()
col_encoding = {}
for col in label_encode_col_list:
    col_encoding[col] = labelencode.fit(process_df[col])
    process_df[col] = col_encoding[col].transform(process_df[col])

sub_df = process_df.loc[:ori_sub_df.shape[0]-1, :]
sub_test_df = process_df.loc[ori_sub_df.shape[0]:, :]

# train_col = ['txkey', 'locdt', 'loctm', 'chid', 'cano', 'contp', 'etymd', 'mchno',
#        'acqic', 'mcc', 'conam', 'ecfg', 'insfg', 'iterm', 'bnsfg', 'flam1',
#        'stocn', 'scity', 'stscd', 'ovrlt', 'flbmk', 'hcefg', 'csmcu', 'csmam',
#        'flg_3dsmk']
train_col = ['locdt', 'loctm', 'contp', 'etymd', 'mchno',
       'acqic', 'mcc', 'conam', 'ecfg', 'insfg', 'iterm', 'bnsfg', 'flam1',
       'stocn', 'scity', 'stscd', 'ovrlt', 'flbmk', 'hcefg', 'csmcu', 'csmam',
       'flg_3dsmk', 'new_location']
label_col = ['label']
X_train, X_test, y_train, y_test = train_test_split(sub_df[train_col], sub_df[label_col], test_size=0.2)

# %%
X_train_val = X_train.values
X_test_val = X_test.values
y_train_val = y_train.values.ravel()
y_test_val = y_test.values.ravel()

model = RandomForestClassifier()
model.fit(X_train_val, y_train_val)

pred = model.predict(X_train_val)
result = score(y_train_val, pred, f'[train]')

pred_test = model.predict(X_test_val)
result = score(y_test_val, pred_test, f'[val]')

data = sub_test_df[train_col].values
ans = sub_test_df['label'].values
general_pred = model.predict(data)
result = score(ans, general_pred, f'[val]')
# %%
