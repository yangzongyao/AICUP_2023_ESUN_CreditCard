# %%
import pandas as pd
from Utils.connect_to_mongo import Mongo
import matplotlib.pyplot as plt

# df = pd.read_csv("Dataset/dataset_1st/training.csv", nrows=100000)
df = pd.read_csv("Dataset/dataset_1st/training.csv")

# %%
label_df = df.groupby("label").count()
# df.count()
# %%
plt.scatter()