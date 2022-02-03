import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn import preprocessing
NBApoints_data = pd.read_csv("NBApoints.csv")
#TODO

Pos = NBApoints_data['Pos']
Tm = NBApoints_data['Tm']
label_encoder_conver = preprocessing.LabelEncoder()
Pos_encoder_value = label_encoder_conver.fit_transform(Pos)
print(Pos_encoder_value)
print("\n")

label_encoder_conver = preprocessing.LabelEncoder()
Tm_encoder_value = label_encoder_conver.fit_transform(Tm)
print(Tm_encoder_value)

train_y = NBApoints_data["3P"]
train_X = pd.DataFrame([Pos_encoder_value,NBApoints_data['Age'], Tm_encoder_value]).T
NBApoints_linear_model = LinearRegression()
NBApoints_linear_model.fit(train_X, train_y)
y_pred = NBApoints_linear_model.predict(train_X)

#NBApoints_linear_model_predict_result=
test_data = [[5,28,10]]
NBApoints_linear_model_predict_result = NBApoints_linear_model.predict(train_X)
print("三分球得球數=",NBApoints_linear_model_predict_result)

print("三分球預測得球數=",NBApoints_linear_model.predict(test_data))

from sklearn.metrics import r2_score
r_squared = r2_score(train_y,y_pred)
print("R_squared值=",r_squared)

print("f_regresstion\n")
(F,p_value) = f_regression(train_X,train_y)
Column_P = pd.DataFrame(f_regression(train_X, NBApoints_data['3P'])[1],train_X.columns)
print("P值=\n",p_value)
print("\n")
print("Pos的 P-value （P值）是否小於 0.05（信心水準 95%）",p_value[0]<0.05)
print("Age的 P-value （P值）是否小於 0.05（信心水準 95%）",p_value[1]<0.05)
