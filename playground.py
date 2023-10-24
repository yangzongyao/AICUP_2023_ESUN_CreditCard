# %%
import pandas as pd
from Utils.connect_to_mongo import Mongo
import matplotlib.pyplot as plt
import pickle

from IPython.display import display

# data process
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.utils import resample

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

from Utils.model_utils import score

# target_col = ['stocn', 'ecfg', 'mcc', 'flam1', 'label']

# with open('./Dataset/data.pkl', 'rb') as f:
#     df = pickle.load(f)

data_num = []
train_score_list = []
test_score_list = []
# data_num_list = range(0, 50, 10)
data_num_list = [50]
for num in data_num_list:
    if num == 0:
        num = 1
    df = pd.DataFrame()
    for i in np.random.randint(0, 398, num):
    # for i in range(0, 399):
        if df.empty:
            df = pd.read_csv(f"./TrainingData_group/group_data{i}.csv")
        else:
            df = pd.concat([df, pd.read_csv(f"./TrainingData_group/group_data{i}.csv")])
            df = df.drop_duplicates()
        
    df = df.drop_duplicates()
    df = resample(df, n_samples=int(df.shape[0]*0.1))
    # general_test_df = pd.read_csv("./TrainingData_group/general_test_data.csv")

    with open('./Dataset/general_test_data.pkl', 'rb') as f:
        general_test_df = pickle.load(f)

    sub_test_df = resample(general_test_df, n_samples=int(general_test_df.shape[0]*1))
    public_data = pd.read_csv("./Dataset/dataset_1st/public_processed.csv")
    ori = public_data.copy()
    # general_test_df = pd.read_csv("./TrainingData_group/group_data2.csv")

    process_df = pd.concat([df, sub_test_df, public_data]).reset_index(drop=True)
    # process_df = pd.concat([df, sub_test_df]).reset_index(drop=True)
    # process_df = process_df[target_col].copy()

    columns_df = pd.read_csv("./Dataset/31_資料欄位說明.csv")
    category_columns = list(columns_df[columns_df["資料格式"] == "類別型"]["訓練資料欄位名稱"].unique())
    category_columns.remove('pred')
    category_columns.remove('label')


    # label encode
    label_encode_col_list = category_columns
    # label_encode_col_list = ['chid', 'cano','acqic','txkey', 'insfg', 'bnsfg', 'mchno', 'scity', 'stscd', 'ovrlt', 'flbmk', 'hcefg', 'csmcu', 'flg_3dsmk']
    labelencode = LabelEncoder()
    col_encoding = {}
    for col in label_encode_col_list:
        col_encoding[col] = labelencode.fit(process_df[col])
        process_df[col] = col_encoding[col].transform(process_df[col])

    # norm 
    # from sklearn.preprocessing import MinMaxScaler
    # norm = MinMaxScaler()
    # process_df = pd.DataFrame(norm.fit_transform(process_df), columns=process_df.columns)

    # one hot encode
    # one_hot_encode_col_list = ['ecfg', 'contp', 'etymd', 'mcc', 'stocn']
    # other_list = list(set(process_df.columns) - set(one_hot_encode_col_list))
    # for col in other_list:
    #     process_df[col] = process_df[col].astype(float)
    # for col in one_hot_encode_col_list:
    #     process_df[col] = process_df[col].astype(str)

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
    X_train_val = X_train.values
    X_test_val = X_test.values
    y_train_val = y_train.values
    y_test_val = y_test.values

    # model_list = [LogisticRegression(), SVC(), DecisionTreeClassifier(),RandomForestClassifier()]
    # model_list = [LogisticRegression()]
    # model_list = [SVC()]
    # model_list = [DecisionTreeClassifier()]
    model_list = [RandomForestClassifier()]

    for model in model_list:
        print(f'------{model.__class__.__name__}-------')
        model.fit(X_train_val, y_train_val)

        pred = model.predict(X_train_val)
        score(y_train_val, pred, f'[train]')
                
        pred_test = model.predict(X_test_val)
        score(y_test_val, pred_test, f'[val]')
                
        # use_col = list(sub_test_df.columns)
        use_col = target_col
        data = sub_test_df[use_col].values
        ans = sub_test_df['label'].values
        
        general_pred = model.predict(data)
        score(ans, general_pred, f'[general test]')

    data_num.append(df.shape[0])
    _, _, train_score = score(y_train_val, pred, f'[train]')
    _, _, test_score = score(ans, general_pred, f'[general test]')
    train_score_list.append(train_score)
    test_score_list.append(test_score)

# plt.plot(data_num, train_score_list)
# plt.plot(data_num, test_score_list)
# plt.legend()
# plt.show()
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


# result_df.to_csv("submit_1577k.csv", index=False)
# %%
