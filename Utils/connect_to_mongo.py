# %%
import pymongo
from urllib.parse import quote
import pandas as pd

class Mongo:
    def __init__(self, host, user, password, port, db) -> None:
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.db = db

    def _connect2mongo(self) -> pymongo.MongoClient:
        client_str = self.convert_client_str(self.host, self.user, self.password, self.port, self.db)
        client = pymongo.MongoClient(client_str)
        return client

    def mongo2dataframe(self, pipe, table_name, id=False) -> pd.DataFrame:
        with self._connect2mongo() as client:
            collection = client[self.db][table_name]
            data_pipe = collection.aggregate(pipe)

        data = [data for data in data_pipe]
        df = pd.DataFrame(data)
        if not id:
            del df["_id"]

        return df
    
    def dataframe2mongo(self, df, table_name) -> bool:
        try:
            with self._connect2mongo() as client:
                collection = client[self.db][table_name]
                collection.insert_many(df.to_dict('records'))
        except:
            return False
        return True
        
    def convert_client_str(self, host, user, password, port, db) -> str:
        user = self.user
        password = quote(self.password)
        host = self.host
        port = self.port
        db = self.db

        client_str = f"mongodb://{user}:{password}@{host}:{port}/{db}"
        return client_str
# %%
if __name__ == '__main__':
    user = "admin"
    password = quote("password")
    host = "0.tcp.jp.ngrok.io"
    port = "15432"
    db = "AstockI_DB"
    client_str = f"mongodb://{user}:{password}@{host}:{port}/{db}"
    client = pymongo.MongoClient(client_str)
    # %%
    self = Mongo(host, user, password, port, db)
    pipe = [{"$match":{"user":"andy"}}]
    df = self.mongo2dataframe(pipe, table_name="testCollection", id=True)
    # %%
    self = Mongo(host, user, password, port, db)
    df = pd.DataFrame([
            {"user":"andy", "mail":"test_mail"}, 
            {"user":"Jim", "mail":"test_mail"},
            {"user":"Bruce", "mail":"test_mail"}
        ])
    self.dataframe2mongo(df, table_name="testCollection")
# %%
