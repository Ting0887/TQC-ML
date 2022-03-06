#from sklearn import datasets
#from sklearn.model_selection import cross_val_predict
#from sklearn import linear_model
# TODO
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import math
boston = load_boston()
# MEDV即預測目標向量
# TODO
X = boston.data
y = boston.target
#分出20%的資料作為test set
# TODO
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)
#Fit linear model 配適線性模型
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)
# TODO
print('MAE:',mean_absolute_error(y_test,y_pred))
print('MSE:',mean_squared_error(y_test,y_pred))
print('RMSE:',math.sqrt(mean_squared_error(y_test,y_pred)))

#  ([[0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30, 396.90 , 4.98]])
X_new = [[0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30, 396.90 , 4.98]]
prediction = lm.predict(X_new)
print(prediction)
