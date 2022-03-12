#使用するライブラリ
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#データの読み込み
train = pd.read_csv("titanic/train.csv")
test = pd.read_csv("titanic/test.csv")

#indexを"PassengerId"に設定
train = train.set_index("PassengerId")
test = test.set_index("PassengerId")

#train, testデータの結合
df = pd.concat([train, test], axis=0, sort=False)

#データの概要確認
df.info()

#欠損値確認
df.isnull().sum()

# Sex
#"Sex"ラベルエンコーディング
df["Sex"] = df["Sex"].map({"female":1, "male":0})

#相関関係を調査
fig, axs = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(),annot=True)
plt.show()

# Embarked
#"Embarked"ラベルエンコーディング
#set(df["Embarked"]) {'C', 'Q', 'S', nan}
df["Embarked"] = df["Embarked"].map({"C":0, "Q":1, "S":2})
#"Embarked"欠損値補完
df["Embarked"] = df["Embarked"].fillna(df.Embarked.median())

#Age
#"Age"欠損値補完
df["Age"] = df["Age"].fillna(df.groupby(["Pclass","Sex"])["Age"].transform("mean"))

#"Age"可視化
fig, axes = plt.subplots(figsize=(10, 5))
sns.histplot(data=df, x="Age")
sns.despine()

#4分割
cut_Age = pd.cut(df["Age"], 4)

#"Survived"との比較
fig, axs = plt.subplots()
sns.countplot(x=cut_Age, hue="Survived", data=df)
sns.despine()

#"Age"ラベルエンコーディング
df['Age'] = LabelEncoder().fit_transform(cut_Age) 

#pandasからグラフ表示（割合）
cross_Age = pd.crosstab(df["Age"], df["Survived"], normalize='index')
cross_Age.plot.bar(figsize=(10, 5)) #stacked=True

#Fare
#"Fare"欠損値補完
df["Fare"] = df["Fare"].fillna(df.groupby(["Pclass", "Sex"])["Fare"].transform("median"))

#4分割
cut_Fare= pd.cut(df["Fare"],4)

#"Survived"との比較
fig, axes = plt.subplots(figsize=(15, 5))
sns.countplot(x=cut_Fare, hue="Survived", data=df)
sns.despine()

#"Fare"ラベルエンコーディング
df["Fare"] = LabelEncoder().fit_transform(cut_Fare) 

#pandasからグラフ表示（割合）
cross_Age = pd.crosstab(df["Fare"], df["Survived"], normalize='index')
cross_Age.plot.bar(figsize=(10, 5)) #stacked=True

#Cabin
#"Cabin"の欠損値補完と数値化
df["Cabin"] = df["Cabin"].apply(lambda x: str(x)[0])
set(df["Cabin"]) #{'A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'n'}

#"Cabin"ごとの"Survived"を確認
df.groupby(df["Cabin"])["Survived"].agg(["mean", "count"])

#"Cabin"ラベルエンコーディング
df["Cabin"] = LabelEncoder().fit_transform(df["Cabin"]) 

#Title
#敬称の種類確認
df["Title"] = df.Name.str.extract("([A-Za-z]+)\.", expand = False)
df["Title"].value_counts()

#敬称を4種類に
other = ["Rev","Dr","Major", "Col", "Capt","Jonkheer","Countess"]

df["Title"] = df["Title"].replace(["Ms", "Mlle","Mme","Lady"], "Miss")
df["Title"] = df["Title"].replace(["Countess","Dona"], "Mrs")
df["Title"] = df["Title"].replace(["Don","Sir"], "Mr")
df["Title"] = df["Title"].replace(other,"Other")

#敬称ごとの生存率を確認
df.groupby("Title").mean()["Survived"]

#敬称ごとの生存関係をグラフ化
fig, axs = plt.subplots(figsize=(15, 5))
sns.countplot(x="Title", hue="Survived", data=df)
sns.despine()

#"Title"ラベルエンコーディング
df["Title"] = LabelEncoder().fit_transform(df["Title"]) 
#{"Mr":2, "Mrs":3, "Miss":1, "Master":0, "Other":4}　辞書順

#pandasからグラフ表示（割合）
cross_Age = pd.crosstab(df["Title"], df["Survived"], normalize='index')
cross_Age.plot.bar(figsize=(10, 5)) #stacked=True

#Family_size
#"Family_size"作成
df["Family_size"] = df["SibSp"] + df["Parch"]+1
#"SibSp", "Parch"をDataFrameから削除
df = df.drop(["SibSp","Parch"], axis = 1)

#家族数ごとの生存関係をグラフ化
fig, axs = plt.subplots(figsize=(15, 5))
sns.countplot(x="Family_size", hue="Survived", data=df)
sns.despine()

#"Family_size"ラベルエンコーディング
df.loc[ df["Family_size"] == 1, "Family_size"] = 0                            #独り身
df.loc[(df["Family_size"] > 1) & (df["Family_size"] <= 4), "Family_size"] = 1  #小家族
df.loc[(df["Family_size"] > 4) & (df["Family_size"] <= 6), "Family_size"] = 2  #中家族
df.loc[df["Family_size"]  > 6, "Family_size"] = 3                             #大家族

#ラベルエンコーディング後、改めてグラフ化
fig, axs = plt.subplots(figsize=(15, 5))
sns.countplot(x="Family_size", hue="Survived", data=df)
sns.despine() 

#pandasからグラフ表示（割合）
cross_Age = pd.crosstab(df["Family_size"], df["Survived"], normalize='index')
cross_Age.plot.bar(figsize=(10, 5)) #stacked=True

#Ticket
#"Ticket"の数字の部分のみ取得
df["Ticket"] = df.Ticket.str.split().apply(lambda x : 0 if x[:][-1] == "LINE" else x[:][-1])
df.Ticket = df.Ticket.values.astype("int64")

#3つの変数をグループ分けして生存率と、生と死を合わせた総人数を調査
s_mean = df.rename(columns={"Survived" : "S_mean"})
s_count = df.rename(columns={"Survived" : "S_count"})
s_mean = s_mean.groupby(["Sex", "Age", "Family_size"]).mean()["S_mean"]
s_count = s_count.groupby(["Sex", "Age", "Family_size"]).count()["S_count"]
pd.concat([s_mean, s_count], axis=1)

#4つの変数をグループ分けして生存率と、生と死を合わせた総人数を調査（男性）
m_s_mean = df.rename(columns={"Survived" : "S_mean"})
m_s_count = df.rename(columns={"Survived" : "S_count"})
m_s_mean = m_s_mean.groupby(["Sex", "Age", "Family_size", "Pclass"]).mean().head(29)["S_mean"]
m_s_count = m_s_count.groupby(["Sex", "Age", "Family_size", "Pclass"]).count().head(29)["S_count"]
pd.concat([m_s_mean, m_s_count], axis=1)

#4つの変数をグループ分けして生存率と、生と死を合わせた総人数を調査（女性）
w_s_mean = df.rename(columns={"Survived" : "S_mean"})
w_s_count = df.rename(columns={"Survived" : "S_count"})
w_s_mean = w_s_mean.groupby(["Sex", "Age", "Family_size", "Pclass"]).mean().tail(31)["S_mean"]
w_s_count = w_s_count.groupby(["Sex", "Age", "Family_size", "Pclass"]).count().tail(31)["S_count"]
pd.concat([w_s_mean, w_s_count], axis=1)

#F_S_Suc
#女性または子どもの家族の生存率を表す説明変数"F_S_Suc"作成
#"Name"の最初を取得
df["TopName"] = df["Name"].map(lambda name:name.split(",")[0].strip())

#女性または子どもはTrue
df["W_C"] = ((df.Title == 0) | (df.Sex == 1))
#女性または子ども以外はTrue
df["M"] = ~((df.Title == 0) | (df.Sex == 1))

#具体的な家族の生存データ
family = df.groupby(["TopName", "Pclass"])["Survived"]

df["F_Total"] = family.transform(lambda s: s.fillna(0).count())
df["F_Total"] = df["F_Total"].mask(df["W_C"], (df["F_Total"] - 1), axis=0)
df["F_Total"] = df["F_Total"].mask(df["M"], (df["F_Total"] - 1), axis=0)

df["F_Survived"] = family.transform(lambda s: s.fillna(0).sum())
df["F_Survived"] = df["F_Survived"].mask(df["W_C"], df["F_Survived"] - df["Survived"].fillna(0), axis=0)
df["F_Survived"] = df["F_Survived"].mask(df["M"], df["F_Survived"] - df["Survived"].fillna(0), axis=0)

df["F_S_Suc"] = (df["F_Survived"] / df["F_Total"].replace(0, np.nan))
df["F_S_Suc"].fillna(-1, inplace = True)

#女性または子ども(True)とそれ以外の人(False)の生存率と生と死を合わせた総人数を調査（家族の生存率ごと）
s_df = df.groupby(["F_S_Suc", "W_C"])["Survived"].agg(["mean", "count"])
s_df

#"F_S_Suc"の計算で使用した説明変数の削除
df.drop(["TopName", "W_C", "M", "F_Total","F_Survived"], axis = 1, inplace = True)

#最後の前処理
#欠損値の確認
df.isnull().sum()

#説明変数選別
df["PassengerId"] = df.index
df.drop(["Name","Embarked","Title", "Cabin"], axis=1, inplace=True)

#ダミー変数化
df = pd.get_dummies(df, columns=["Sex"], drop_first=True)
df = pd.get_dummies(df, columns=["Pclass", "Fare"])

#"Ticket"のみ標準化
num_features = ["Ticket"]
for col in num_features:
    scaler = StandardScaler()
    df[col] = scaler.fit_transform(np.array(df[col].values).reshape(-1, 1))

#元の形に戻す（train, testデータの形に）
train, test = df.loc[train.index], df.loc[test.index]
#学習用データ
x_train = train.drop(["PassengerId","Survived"], axis = 1)
y_train = train["Survived"]
train_names = x_train.columns
#テスト用データ
x_test = test.drop(["PassengerId","Survived"], axis = 1)

#モデル構築
#決定木
decision_tree = DecisionTreeClassifier(random_state=0, max_depth=3)
#学習
decision_tree.fit(x_train, y_train)
#推論
y_pred = decision_tree.predict(x_train)

#正解率： 0.8125701459034792
print("正解率：", accuracy_score(y_train, y_pred))

#提出データ1
y_pred = decision_tree.predict(x_test)

#説明変数の重要度をグラフで表示（決定木）
importances = pd.DataFrame(decision_tree.feature_importances_, index = train_names)
importances.sort_values(by = 0, inplace=True, ascending = False)
importances = importances.iloc[0:6,:] 
plt.figure(figsize=(8, 5)) 
sns.barplot(x=0, y=importances.index, data=importances,palette="deep").set_title("Feature Importances",
                                                                                 fontdict= { 'fontsize': 20,
                                                                                            'fontweight':'bold'});
sns.despine()

#xgboost
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)
#パラメータ
params = {'colsample_bytree': 0.5, 
         'learning_rate': 0.1, 
         'max_depth': 3, 
         'subsample': 0.9, 
         "objective":"multi:softmax", 
         "num_class":2}
#学習
bst = xgb.train(
    params, 
    dtrain, 
    num_boost_round=10)

#推論
y_pred_2 = bst.predict(dtrain)

#正解率： 0.8215488215488216
print("正解率：",accuracy_score(y_train, y_pred_2))

#提出データ2
y_pred_2 = bst.predict(dtest)

#説明変数の重要度をグラフで表示（xgboost）
fig, ax = plt.subplots(figsize=(12, 4))
"""
mapper = {'f{0}'.format(i): v for i, v in enumerate(train_names)}
mapped = {mapper[k]: v for k, v in bst.get_score(importance_type="gain").items()}

xgb.plot_importance(mapped,
                    ax=ax, 
                    show_values=False)
"""
xgb.plot_importance(bst,
                    ax=ax, 
                    show_values=False, 
                    importance_type="gain")
plt.show()

#submit用のファイル１を作成(決定木)
submit = pd.DataFrame({"PassengerId":test["PassengerId"], "Survived":y_pred.astype(int).ravel()})
submit.to_csv("answer_xgb_2021_06_24.csv",index = False)

#submit用のファイル２を作成(xgboost)
submit = pd.DataFrame({"PassengerId":test["PassengerId"], "Survived":y_pred_2.astype(int).ravel()})
submit.to_csv("answer_tree_2021_06_24.csv",index = False)
