# AICUP_2023_CreditCard

# dataset
Mongo DB
```
user = "admin"
password = "password"
host = "0.tcp.jp.ngrok.io"
port = "18169"
db = "AICUP"
table_name="ESUN_TrainingData"
```

## Resource
* [T-Brain](https://tbrain.trendmicro.com.tw/Competitions/Details/31)
* [AI CUP-TEAM_4201](https://go.aicup.tw/competition/team/aa9d73cf-97aa-4be2-8775-7cbc68b11cf9/)

# submit log:
- test
### log1: LabelEncoder + LogisticRegression
|Advanced_Public_Recall|Advanced_Public_Precision|score|
|----|---|---|
|0.950549|0.004214|0.008391|
---
### log2: LabelEncoder + RandomForestClassifier
- 重新分割資料集
- train、test、submit的類別欄位，一起encoding
- 捨棄null欄位(stscd)

|Advanced_Public_Recall|Advanced_Public_Precision|score|
|----|---|---|
|0.924451|0.038346|0.073638|