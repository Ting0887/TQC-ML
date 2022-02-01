# TODO
from sklearn.datasets import load_iris
iris_dataset = load_iris()

# TODO
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# create dataframe from data in X_train 根據X_train中的資料創建dataframe
# label the columns using the strings in iris_dataset.feature_names 使用iris_dataset.feature_names中的字串標記列
# TODO
X = iris_dataset.data
y = iris_dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4,random_state=1)
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print("Test set score: ",round(accuracy_score(y_test, y_pred),4)          )

# TODO
pred_test_1 = knn.predict(np.array([[5, 2.9, 1, 0.2]]))
print("Predicted target name:",iris_dataset.target_names[pred_test_1])

# TODO
pred_test_2 = knn.predict(np.array([[5.7, 2.8, 4.5,1.2]]))
print("Predicted target name:",iris_dataset.target_names[pred_test_2])

# TODO
pred_test_3 = knn.predict(np.array([[7.7, 3.8, 6.7, 2.1]]))
print("Predicted target name:",iris_dataset.target_names[pred_test_3])
