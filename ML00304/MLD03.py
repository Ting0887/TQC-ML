from sklearn import datasets
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
# TODO
diabete = load_diabetes()

#get x
# TODO 
X = diabete.data
y = diabete.target
print(X)
#Total number of examples
# TODO 
lm = LinearRegression()
lm.fit(X, y)
y_pred = lm.predict(X)
print('Total number of examples')
print('MSE=',round(mean_squared_error(y, y_pred),4))
print('R-squared=',round(r2_score(y, y_pred),4))
#3:1 100
xTrain2, xTest2, yTrain2, yTest2= train_test_split(X, y,test_size=0.25,random_state=100)
lm2=LinearRegression()
lm2.fit(xTrain2,yTrain2)
# TODO 
y_pred2 = lm2.predict(xTest2)
y_train2 = lm2.predict(xTrain2)

print('Split 3:1')
print('train MSE=',round(mean_squared_error(yTrain2, y_train2),4))
print('test MSE=',round(mean_squared_error(yTest2, y_pred2),4))
print('train R-squared=',round(r2_score(yTrain2, y_train2),4))
print('test R-squared=',round(r2_score(yTest2,y_pred2),4))
