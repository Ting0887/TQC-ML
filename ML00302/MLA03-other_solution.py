#from sklearn import datasets
#from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
# TODO
import pandas as pd
from sklearn.datasets import load_boston
boston = load_boston()
df_X = pd.DataFrame(boston.data.T, ['CRIM','ZN','INDUS','CHAS','NOX','RM' ,'AGE','DIS','RAD','TAX', 'PTRATIO','B','LSTAT']).T #有13個feature
df_y = pd.DataFrame(boston.target.T,columns=['MEDV'])
# TODO
# MEDV即預測目標向量
# TODO

X = df_X[['CRIM','ZN','INDUS','CHAS','NOX','RM' ,'AGE','DIS','RAD','TAX', 'PTRATIO','B','LSTAT']]
y = df_y['MEDV']

#分出20%的資料作為test set
# TODO
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)
#Fit linear model 配適線性模型
lm = linear_model.LinearRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)
from sklearn.metrics import mean_absolute_error,mean_squared_error
import math
# TODO
print('MAE:',round(mean_absolute_error(y_test, y_pred), 4))
print('MSE:',round(mean_squared_error(y_test, y_pred), 4))
print('RMSE:',round(math.sqrt(mean_squared_error(y_test, y_pred)),4))

#  ([[0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30, 396.90 , 4.98]])
X_new = [[0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30, 396.90 , 4.98]]
prediction = lm.predict(X_new)
print(prediction)
