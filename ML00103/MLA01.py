import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
#import matplotlib.pyplot as plt
#注意匯入
import math
import warnings
warnings.filterwarnings("ignore")

input_file = 'cardata.txt'

df = pd.read_csv(input_file,header = None)
# 注意要複製一份出來
df_source = df.copy(deep=False)
# Reading the data
# X = df.iloc[:,0:df.shape[1]-1]
# y = df.iloc[:,df.shape[1]
# TODO
print(df)

# Convert string data to numerical data將字串資料轉換為數值資料
# TODO
label_encoder = preprocessing.LabelEncoder()
for name in df.columns:
    # 注意二維陣列
    df_source[name] = label_encoder.fit_transform(df_source[[name]])


# Build a Random Forest classifier建立隨機森林分類器
# TODO
# 注意要名稱為classifier及參數設定
classifier = RandomForestClassifier(n_estimators=200,max_depth=8,random_state=7)
# 注意取值範圍
X = df_source.iloc[:,0:6]
y = df_source.iloc[:,-1]
classifier.fit(X,y)

# # Cross validation交叉驗證
from sklearn import model_selection
# # TODO
val = model_selection.cross_validate(classifier,X,y,cv=3)


print("Accuracy of the classifier=" + \
      '{0:.2f}'.format(round(val['test_score'].mean()*100,2))       + "%")

# # Testing encoding on single data instance測試單個資料實例上的編碼
input_data = ['high', 'low', '2', 'more', 'med', 'high']
# # TODO
result_data = []
for i in range(len(input_data)):
    # 注意取0
    result_data.append(label_encoder.fit(df.iloc[:,i]).transform([input_data[i]])[0])

y_pred = classifier.predict([result_data])
# car為第六個
namelist = list(label_encoder.fit(df.iloc[:,6]).classes_)
# # Predict and print output for a particular datapoint
# # TODO
print("Output class=",namelist[int(y_pred)])

# ########################
# # Validation curves 驗證曲線
# 注意import
from sklearn.model_selection import validation_curve
# # TODO
parameter_grid = np.linspace(25,200,8).astype(int)

train_scores, validation_scores = validation_curve(classifier, X, y, 
        "n_estimators", parameter_grid, cv=5)
print("##### VALIDATION CURVES #####")
print("\nParam: n_estimators\nTraining scores:\n", train_scores)
print("\nParam: n_estimators\nValidation scores:\n", validation_scores)
# 注意無條件捨去寫法
print("Training scores第一組第一筆數值:",'{:.4f}'.format(\
                                math.floor(train_scores[0][0]*(10**4))/(10**4)))
print("Validation scores最後一組第一筆數值:",'{:.4f}'.format(\
                                math.floor(validation_scores[7][0]*(10**4))/(10**4)))