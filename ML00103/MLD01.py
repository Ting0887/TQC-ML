import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
input_file = 'cardata.txt'

# Reading the data
X = []
y = []
# TODO
count = 0 
with open(input_file, 'r') as f: 
    for line in f.readlines(): 
        data = line[:-1].split(',')
        X.append(data)
X = np.array(X)

# Convert string data to numerical data將字串資料轉換為數值資料
# TODO
label_encoder = [] 
X_encoded = np.empty(X.shape) 
for i,item in enumerate(X[0]): 
    label_encoder.append(preprocessing.LabelEncoder()) 
    X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i]) 
X = X_encoded[:, :-1].astype(int) 
y = X_encoded[:, -1].astype(int)


# Build a Random Forest classifier建立隨機森林分類器
# TODO
params = {"n_estimators":200, "criterion":'entropy', "max_depth":8, "random_state":7} 
classifier = RandomForestClassifier(**params)
classifier.fit(X, y)


# Cross validation交叉驗證
from sklearn import model_selection
# TODO
accuracy = model_selection.cross_val_score(classifier, X, y,scoring='accuracy', cv=3) 
print("Accuracy of the classifier: " + str(round(100*accuracy.mean(), 2)) + "%")

# Testing encoding on single data instance測試單個資料實例上的編碼
input_data = ['high', 'low', '2', 'more', 'med', 'high']
# TODO
input_data_encoded = [-1]*len(input_data) 
for i, item in enumerate(input_data):
    input_data_encoded[i] = int(label_encoder[i].transform([input_data[i]]))


input_data_encoded = np.array(input_data_encoded) 
# 將標記編碼後的單一樣本轉換成numpy數組 
input_data_encoded = input_data_encoded.reshape(1, len(input_data))

# Predict and print output for a particular datapoint
# TODO
output_class = classifier.predict(input_data_encoded)
print("Output class=" ,label_encoder[-1].inverse_transform(output_class)[0]   )

########################
# Validation curves 驗證曲線
classifier = RandomForestClassifier(max_depth=8, random_state=7,n_estimators=200) 
# 固定max_depth參數為4定義分類器 
parameter_grid = np.linspace(25, 200, 8).astype(int) 
# 觀察評估器數量對訓練得分的影響，評估器每8疊代一次 
train_scores, validation_scores = model_selection.validation_curve(classifier, X, y, "n_estimators", parameter_grid, cv=5) 
print("\nParam: n_estimators\nTraining scores:\n", train_scores) 
print("\nParam: n_estimators\nValidation scores:\n", validation_scores)
classifier = RandomForestClassifier(n_estimators=20, random_state=7)
parameter_grid = np.linspace(2, 10, 5).astype(int) 
train_scores, valid_scores = model_selection.validation_curve(classifier, X, y, "max_depth", parameter_grid, cv=5)
print("\nParam: max_depth\nTraining scores:\n", train_scores) 
print("\nParam: max_depth\nValidation scores:\n", validation_scores)

# TODO

parameter_grid = np.linspace(25, 200, 8).astype(int)
train_scores, validation_scores = validation_curve(classifier, X, y, 
        "n_estimators", parameter_grid, cv=5)
print("##### VALIDATION CURVES #####")
print("\nParam: n_estimators\nTraining scores:\n", train_scores)
print("\nParam: n_estimators\nValidation scores:\n", validation_scores)

plt.figure()
plt.plot(parameter_grid, 100*np.average(train_scores, axis=1), color='black') 
plt.title('Validation curve') 
plt.xlabel('Maximum depth of the tree') 
plt.ylabel('Accuracy') 
plt.show()


