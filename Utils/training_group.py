# %%
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from IPython.display import display

with open('../Dataset/data.pkl', 'rb') as f:
    df = pickle.load(f)

df = df.loc[df["stscd"].isna(), :]

training_set, general_test_set = train_test_split(df, test_size=0.2)
display(training_set.groupby("label").count())
display(general_test_set.groupby("label").count())

result = "../TrainingData_group"

fraud_df = training_set[training_set["label"]==1]
normal_df = training_set[training_set["label"]==0]

normal_df_shuffle = shuffle(normal_df)
normal_df_shuffle = normal_df_shuffle.reset_index(drop=True)

n = 0
scope = fraud_df.shape[0]
for n in range(normal_df_shuffle.shape[0] // scope + 1):
    temp_df = normal_df_shuffle.loc[scope*n: scope*(n+1), :]
    result_df = pd.concat([temp_df, fraud_df])
    result_df.to_csv(f"{result}/group_data{n}.csv", index=False)

general_test_set.to_csv(f"{result}/general_test_data.csv", index=False)
# %%
