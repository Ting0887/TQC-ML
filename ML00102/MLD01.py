import pandas as pd
import numpy as np
from sklearn import preprocessing, linear_model

# 原始資料
titanic = pd.read_csv("titanic.csv")
print('raw data')
# TODO
mid_age = titanic['Age'].median()

# 將年齡的空值填入年齡的中位數
# TODO
age_fillna = np.where(titanic['Age'].isnull(),mid_age,titanic['Age'])
titanic['Age'] = age_fillna
print("年齡中位數="        ,mid_age)
# TODO

# 更新後資料
print('new data')
# TODO

# 轉換欄位值成為數值
label_encoder = preprocessing.LabelEncoder()
encoded_class = label_encoder.fit_transform(titanic['PClass'])
X = pd.DataFrame([encoded_class,titanic['Age'],titanic['SexCode']])
X = X.T
y = titanic['Survived']
# TODO


# 建立模型
# TODO
model = linear_model.LogisticRegression()
model.fit(X,y)

print('截距=',model.intercept_)
print('迴歸係數=',model.coef_)


# 混淆矩陣(Confusion Matrix)，計算準確度
print('Confusion Matrix')
# TODO
pred = model.predict(X)
print(pd.crosstab(pred,y))
print(model.score(X, y))



