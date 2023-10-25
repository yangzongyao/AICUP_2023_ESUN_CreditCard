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

ori_sub_df = df[~df["stscd"].isna()].copy()
# concat_sub_df = resample(df[df["stscd"].isna()], n_samples=ori_sub_df.shape[0])

# process_df = pd.concat([ori_sub_df, concat_sub_df])
process_df = ori_sub_df

train_col = ['locdt', 'loctm', 'contp', 'etymd', 'mchno',
       'acqic', 'mcc', 'conam', 'ecfg', 'insfg', 'iterm', 'bnsfg', 'flam1',
       'stocn', 'scity', 'ovrlt', 'flbmk', 'hcefg', 'csmcu', 'csmam',
       'flg_3dsmk']

process_df = process_df[train_col + ['stscd']].copy()

# %%
# label encode
# label_encode_col_list = ['txkey', 'chid', 'cano', 'contp', 'etymd', 'mchno', 'acqic', 'mcc', 'ecfg', 'insfg', 'bnsfg', 'stocn', 'scity', 'stscd', 'ovrlt', 'flbmk', 'hcefg', 'csmcu', 'flg_3dsmk']
label_encode_col_list = ['loctm', 'mchno', 'contp', 'etymd', 'acqic', 'mcc', 'ecfg', 'insfg', 'bnsfg', 'stocn', 'scity', 'stscd', 'ovrlt', 'flbmk', 'hcefg', 'csmcu', 'flg_3dsmk']
labelencode = LabelEncoder()
col_encoding = {}
for col in label_encode_col_list:
    col_encoding[col] = labelencode.fit(process_df[col])
    process_df[col] = col_encoding[col].transform(process_df[col])

sub_df = process_df.copy()
# %%
sub_df_train = sub_df.copy()
sub_df_train_ans = sub_df_train['stscd']
del sub_df_train['stscd']
X_train, X_test, y_train, y_test = train_test_split(sub_df_train, sub_df_train_ans, test_size=0.2)

# %%
X_train_val = X_train.values
X_test_val = X_test.values
y_train_val = y_train.values.ravel()
y_test_val = y_test.values.ravel()

model = RandomForestClassifier()
model.fit(X_train_val, y_train_val)

# %%
with open('stscd_null_model.pkl', 'wb') as f:
    pickle.dump(model, f)
# %%
