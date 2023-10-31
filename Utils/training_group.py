# %%
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from IPython.display import display
import numpy as np
from pathlib import Path

with open('./Dataset/data.pkl', 'rb') as f:
    df = pickle.load(f)

# df = df.loc[df["stscd"].isna(), :]
def split_8_2():
    training_set, general_test_set = train_test_split(df, test_size=0.2)
    display(training_set.groupby("label").count())
    display(general_test_set.groupby("label").count())

    result = "./TrainingData_group"

    normal_df_shuffle = shuffle(training_set)
    result_df = normal_df_shuffle.reset_index(drop=True)

    Path(result).mkdir(exist_ok=True, parents=True)
    result_df.to_csv(f"{result}/training_data.csv", index=False)
    general_test_set.to_csv(f"{result}/general_test_data.csv", index=False)
    
    with open(f"{result}/training_data.pkl", 'wb') as f:
        pickle.dump(result_df, f)

    with open(f"{result}/general_test_data.pkl", 'wb') as f:
        pickle.dump(general_test_set, f)

def split_simulation_distributed():
    # %%
    result = "./TrainData_simulation_distributed_group"
    # non_null_df = df[df["stscd"].isna()]
    normal_df = df[df["label"]==0].copy()
    fraud_df = df[df["label"]==1].copy()
    fraud_09, fraud_01 = train_test_split(fraud_df, test_size=0.1)
    ratio_of_fraud = normal_df.shape[0] / fraud_df.shape[0]
    normal_01_count = fraud_01.shape[0] * ratio_of_fraud
    normal_01 = normal_df.sample(n=int(normal_01_count), random_state=1)
    general_test_set = pd.concat([normal_01, fraud_01])
    
    other_index = list(set(df.index) - set(general_test_set.index))
    other_df = df.iloc[other_index]

    # other_fraud = other_df[other_df["label"]==1]
    # n = 0
    # scope = other_fraud.shape[0]
    # for n in range(normal_df_shuffle.shape[0] // scope + 1):
    #     temp_df = normal_df_shuffle.loc[scope*n: scope*(n+1), :]
    #     result_df = pd.concat([temp_df, fraud_df])
    #     result_df.to_csv(f"{result}/group_data{n}_stscd.csv", index=False)

    other_df.to_csv(f"{result}/training_data.csv", index=False)
    general_test_set.to_csv(f"{result}/general_test_data.csv", index=False)

    with open(f"{result}/training_data.pkl", 'wb') as f:
        pickle.dump(other_df, f)

    with open(f"{result}/general_test_data.pkl", 'wb') as f:
        pickle.dump(general_test_set, f)

# %%
