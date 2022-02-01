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

train_X = pd.DataFrame([Pos_encoder_value,NBApoints_data['Age'], Tm_encoder_value]).T
NBApoints_linear_model = LinearRegression()
NBApoints_linear_model.fit(train_X, NBApoints_data["3P"])

#NBApoints_linear_model_predict_result=
test_data = np.array([5,28,10]).reshape(1,-1)
NBApoints_linear_model_predict_result = NBApoints_linear_model.predict(test_data)
print("三分球得球數=",NBApoints_linear_model_predict_result)

r_squared = NBApoints_linear_model.score(train_X, NBApoints_data["3P"])
print("R_squared值=",r_squared)

print("f_regresstion\n")
Column_P = pd.DataFrame(f_regression(train_X, NBApoints_data['3P'])[1],train_X.columns)
print("P值=\n",Column_P[0])
print("\n")
