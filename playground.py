# %%
import pandas as pd
from Utils.connect_to_mongo import Mongo

df = pd.read_csv("Dataset/dataset_1st/training.csv")

# %%
user = "admin"
password = "password"
host = "0.tcp.jp.ngrok.io"
port = "19977"
db = "AICUP"
mongo = Mongo(host, user, password, port, db)
mongo.dataframe2mongo(df, table_name="ESUN_TrainingData")
# %%
df.describe()
# %%
