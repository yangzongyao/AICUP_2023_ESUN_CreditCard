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
---
### log3: LabelEncoder + RandomForestClassifier
- 放大訓練集(200k)

|Advanced_Public_Recall|Advanced_Public_Precision|score|
|----|---|---|
|0.614011|0.167458|0.263148|
---
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

## note

loctm 當作刷卡時間，看能不能區分白天or晚上...等等
csmam 可能要跟幣別掛在一起
conam 可能可以完全取代 csmam

insfg(是否分期) 跟 iterm(分期期數) 可以一起討論，也可以當作交乘項來試試看 


0.45 => 盜刷資料切開留9：1，Training 30:1 的比例，val分數0.51， 遺失欄位補值(用類似的補)

val 8:2 => 不要把2的再拆小，放全部下去測測看

看那個國家的盜刷特別多，把那個國家獨立成一個類別


## process note
txkey 欄位為一個類別一筆資料
locdt 授權日期平均分散
loctm 為授權時間，按照時分秒排序，看能不能轉成早晚之類的類別使用

chid : 01ec8ef3f97d04ad36a94e1bea5464bd45f8026a2fefef6cf6fbd660c3884f04
交易間隔很短，幣值不為台幣，且扣除掉txkey後大量重複

loctm 計算時間並轉換成類別
stscd 狀態碼為1的留著，其餘更改為0
chid encode為刷卡次數
分析ecfg網路交易欄位


應該把Test PUBLIC Train的LabelEncoding一起編


log 5:



note:

loctm : 4、2小時各切一個比較
再產白天、晚上的欄位

匯率： 直接補匯率



csmcu 和 csmam為null跟0 特定hcefg支付型態下有大量的1


t t+1 時間欄位問題

chid_fraud 欄位會有t 跟t+1的問題，maybe會影響val跟test的分數差異很大