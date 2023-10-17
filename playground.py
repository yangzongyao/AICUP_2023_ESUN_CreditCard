# %%
import pandas as pd
from Utils.connect_to_mongo import Mongo
import matplotlib.pyplot as plt
import pickle

from IPython.display import display

# data process
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.utils import resample

from Utils.model_utils import score

# target_col = ['stocn', 'ecfg', 'mcc', 'flam1', 'label']

# with open('./Dataset/data.pkl', 'rb') as f:
#     df = pickle.load(f)
df = pd.read_csv("./TrainingData_group/group_data0.csv")
# df = pd.concat([df, pd.read_csv("./TrainingData_group/group_data1.csv")])
# df = pd.concat([df, pd.read_csv("./TrainingData_group/group_data2.csv")])

general_test_df = pd.read_csv("./TrainingData_group/general_test_data.csv")
sub_test_df = resample(general_test_df, n_samples=int(general_test_df.shape[0]*0.1))

public_data = pd.read_csv("./Dataset/dataset_1st/public_processed.csv")
ori = public_data.copy()
# general_test_df = pd.read_csv("./TrainingData_group/group_data2.csv")

process_df = pd.concat([df, sub_test_df, public_data]).reset_index(drop=True)
# process_df = process_df[target_col].copy()

columns_df = pd.read_csv("./Dataset/31_資料欄位說明.csv")
category_columns = list(columns_df[columns_df["資料格式"] == "類別型"]["訓練資料欄位名稱"].unique())
category_columns.remove('pred')
category_columns.remove('label')

# label encode
labelencode = LabelEncoder()
col_encoding = {}
for col in category_columns:
    col_encoding[col] = labelencode.fit(process_df[col])
    process_df[col] = col_encoding[col].transform(process_df[col])

# norm 
# from sklearn.preprocessing import MinMaxScaler
# norm = MinMaxScaler()
# process_df = pd.DataFrame(norm.fit_transform(process_df), columns=process_df.columns)

# one hot encode
# process_df["stocn"] = process_df["stocn"].astype(str)
# process_df["ecfg"] = process_df["ecfg"].astype(str)
# process_df["mcc"] = process_df["mcc"].astype(str)
# process_df = pd.get_dummies(process_df)

df = process_df.loc[:df.shape[0]-1, :].copy()
sub_test_df = process_df.loc[df.shape[0]:df.shape[0]+sub_test_df.shape[0]-1, :].copy()
public_data = process_df.loc[df.shape[0]+sub_test_df.shape[0]:, :].copy()

df = shuffle(df)

# display(df.corr().sort_values('label', ascending=False)["label"])

target_col = list(df.columns)
# top_n = 6
# corr_list = df.corr().sort_values('label', ascending=False)["label"]
# target_col = corr_list[:top_n].index.to_list()
target_col.remove('label')
X = df[target_col]
y = df['label']

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

# model_list = [LogisticRegression(), SVC(), DecisionTreeClassifier(),RandomForestClassifier()]
model_list = [LogisticRegression()]
# model_list = [SVC()]
# model_list = [DecisionTreeClassifier()]
# model_list = [RandomForestClassifier()]

for model in model_list:
    print(f'------{model.__class__.__name__}-------')
    model.fit(X_train, y_train)

    pred = model.predict(X_train)
    score(y_train, pred, '[train]')

    pred_test = model.predict(X_test)
    score(y_test, pred_test, '[val]')

    # use_col = list(sub_test_df.columns)
    use_col = target_col
    data = sub_test_df[use_col].values
    ans = sub_test_df['label'].values

    general_pred = model.predict(data)
    score(ans, general_pred, '[general test]')

# %%
# save submit.csv
# del public_data["label"]
# X = public_data.values
# public_pred = model.predict(X)

# public_data["pred"] = public_pred.astype(int)

# result_df = pd.DataFrame({
#     'txkey': ori["txkey"].values,
#     'pred': public_pred.astype(int)
# })

# display(result_df.groupby('pred').count())


# result_df.to_csv("submit.csv", index=False)
# %%
