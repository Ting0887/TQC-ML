import csv
from sklearn.model_selection import train_test_split

input_file = 'wine.csv'

X = []
y = []

# TODO
with open('wine.csv','r',encoding='utf-8') as csvf:
    r = csv.reader(csvf)
    for item in r:
        X.append(item[1:]) # data features
        y.append(item[0]) # data labels

from sklearn import model_selection

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=5,test_size=0.25)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# TODO
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
accuracy = round(clf.score(X_test, y_test)*100,2)

# compute accuracy of the classifier計算分類器的精確度
print("Accuracy of the classifier = %.2f"%accuracy, "%")

X_test1 =[[1.51, 1.73, 1.98, 20.15, 85, 2.2, 1.92, .32, 1.48, 2.94, 1, 3.57, 172]]
X_test2 = [[14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, .28, 2.29, 5.64, 1.04, 3.92, 1065]]
X_test3 = [[13.71, 5.65, 2.45, 20.5, 95, 1.68, .61, .52, 1.06, 7.7, .64, 1.74, 720]]


# TODO
print('X_test1分類結果:%d'%clf.predict(X_test1,y_test))
print('X_test2分類結果:%d'%clf.predict(X_test2,y_test))
print('X_test3分類結果:%d'%clf.predict(X_test3,y_test))