# AICUP_2023_CreditCard

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
### log3: LabelEncoder + RandomForestClassifier
- 放大訓練集(200k)

|Advanced_Public_Recall|Advanced_Public_Precision|score|
|----|---|---|
|0.614011|0.167458|0.263148|
### log4: LabelEncoder + RandomForestClassifier
- 放大訓練集(450k)

|Advanced_Public_Recall|Advanced_Public_Precision|score|
|----|---|---|
|0.46978|0.326128|0.384991|


# TO DO:
- 測試訓練集放大的效果
- 測試訓練集放大時 test set 能否反映出 public set分數
- 畫分佈圖，確認欄位重要程度

## Resource
* [T-Brain](https://tbrain.trendmicro.com.tw/Competitions/Details/31)
* [AI CUP-TEAM_4201](https://go.aicup.tw/competition/team/aa9d73cf-97aa-4be2-8775-7cbc68b11cf9/)

