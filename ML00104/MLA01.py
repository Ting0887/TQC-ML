import pandas as pd

# 載入寶可夢資料集
# TODO
df = pd.read_csv('pokemon.csv')
df.info()
# 處理遺漏值
features = ['Attack', 'Defense']
# TODO
df.dropna(axis=0,subset=features,inplace=True)
# 取出目標寶可夢的 Type1 與兩個特徵欄位
# TODO
extract_features = df[(df['Type1'] == 'Normal') |\
                      (df['Type1'] == 'Fighting') |\
                       (df['Type1']=='Ghost')]
X_train, y_train = extract_features[features], extract_features['Type1']
# 編碼 Type1
from sklearn.preprocessing import LabelEncoder
# TODO
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
# 特徵標準化
from sklearn.preprocessing import StandardScaler
# TODO
std = StandardScaler().fit(X_train)
X_train_std = std.transform(X_train)

# 建立線性支援向量分類器，除以下參數設定外，其餘為預設值
# #############################################################################
# C=0.1, dual=False, class_weight='balanced'
# #############################################################################
from sklearn.svm import LinearSVC
# TODO
model = LinearSVC(C=0.1,dual=False,class_weight='balanced')
model.fit(X_train_std,y_train)
# 計算分類錯誤的數量
# TODO
y_pred = model.predict(X_train_std)
print('Missing classified samples: %d' %(y_train!=y_pred).sum())
# 計算準確度(accuracy)
from sklearn.metrics import accuracy_score
print('Accuracy: %.4f' %accuracy_score(y_train, y_pred))

# 計算有加權的 F1-score (weighted)
from sklearn.metrics import f1_score
# TODO
print('F1-score:%.4f '%f1_score(y_train, y_pred,average='weighted'))

# 預測未知寶可夢的 Type1
# TODO
new_data = [[100,75]]
pred_label = label_encoder.inverse_transform(model.predict(new_data))
print('預測分類結果:',pred_label)
