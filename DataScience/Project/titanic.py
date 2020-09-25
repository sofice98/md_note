import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

title2num = {"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Dr":5,"Rev":6,"Mlle":7,"Major":8,
            "Col":9,"Countess":10,"Jonkheer":11,"Sir":12,"Lady":13,"Ms":14,"Capt":15,"Mme":16,"Don":17,"Dona":18}

# 预处理
def processData(data):
    # 填充缺失值
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Embarked"] = data["Embarked"].fillna("S")
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())
    # 编码
    data['Sex'] = data['Sex'].apply(lambda r : 1 if r == "male" else 0)
    data['Embarked'] = data['Embarked'].apply(lambda r : 1 if r == "C" else r)
    data['Embarked'] = data['Embarked'].apply(lambda r : 2 if r == "S" else r)
    data['Embarked'] = data['Embarked'].apply(lambda r : 3 if r == "Q" else r)

    return data

# 特征构建
def featureConstruction(data):
    #data["FamilySize"] = data["SibSp"] + data["Parch"]
    data["NameLength"] = data["Name"].apply(lambda x:len(x))
    data["Title"] = data["Name"].apply(getTitle)

def getTitle(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title2num[title_search.group(1)]
    return 0

# 特征选择
def featureSelection(data):
    predictors = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Title","NameLength"]
    selector = SelectKBest(f_classif, k=5)
    selector.fit(data[predictors], data["Survived"])

    scores = -np.log10(selector.pvalues_)

    plt.bar(range(len(predictors)), scores)
    plt.xticks(range(len(predictors)), predictors, rotation='vertical')
    plt.show()

def featureEngineering(data):
    processData(data)
    featureConstruction(data)
    #featureSelection(data)

# 训练数据
def trainData(data):
    predictors = ["Pclass","Sex","Fare","Title","NameLength"]
    model = RandomForestClassifier(n_estimators=50, min_samples_split=4, min_samples_leaf=2)
    model.fit(data[predictors], data["Survived"])
    # scores = cross_val_score(model, data[predictors], data["Survived"], cv=3)
    # print(scores.mean())
    return model

# 集成学习
# def ensembleLearn(data):
#     algorithms = {
#         [GradientBoostingClassifier(n_estimators=25, max_depth=3), ["Pclass","Sex","Fare","Title","NameLength"]]
#         [LogisticRegression(), ["Pclass","Sex","Fare","Title","NameLength"]]
#     }

def predict(model, data):
    predictors = ["Pclass","Sex","Fare","Title","NameLength"]
    prob = model.predict(data[predictors])
    out = pd.DataFrame({"PassengerId": data["PassengerId"], "Survived": prob}, 
                        columns=['PassengerId', 'Survived'])
    out.to_csv("titanic/out.csv", index = False)

if __name__ == "__main__":
    # 读取数据
    train = pd.read_csv('titanic/train.csv')
    test = pd.read_csv('titanic/test.csv')
    # 特征工程
    featureEngineering(train)
    featureEngineering(test)
    # 训练数据
    model = trainData(train)
    # 预测数据
    predict(model, test)
