# #############################################################################
# 本題參數設定，請勿更改
seed = 0    # 亂數種子數
# #############################################################################

import pandas as pd

# 載入寶可夢資料
# TODO
df = pd.read_csv('pokemon.csv')
# 取出目標欄位
columns = ['Defense','SpecialAtk']
X = df.loc[:,columns] #TODO     特徵欄位
y = df.loc[:,'Type1'] #TODO     Type1 欄位

# 編碼 Type1
from sklearn import preprocessing
# TODO
y_train = preprocessing.LabelEncoder().fit_transform(y)
# 切分訓練集、測試集，除以下參數設定外，其餘為預設值
# #########################################################################
# X, y, test_size=0.2, random_state=seed
# #########################################################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=seed)
X_train_score = X_train.copy(deep=False)
# 特徵標準化
from sklearn.preprocessing import StandardScaler
std = StandardScaler().fit(X_train)
X_train_std = std.fit_transform(X_train)
# 訓練集
# 分別建立 RandomForest, kNN, SVC, Voting，除以下參數設定外，其餘為預設值
# #############################################################################
# RandomForest: n_estimators=10, random_state=seed
# kNN: n_neighbors=4
# SVC: gamma=.1, kernel='rbf', probability=True
# Voting: estimators=[('RF', clf1), ('kNN', clf2), ('SVC', clf3)], 
#         voting='hard', n_jobs=-1
# #############################################################################    
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# TODO
clf1 = RandomForestClassifier(n_estimators=10,random_state=seed)
clf2 = KNeighborsClassifier(n_neighbors=4)
clf3 = SVC(gamma=.1,kernel='rbf',probability=True)
vote = VotingClassifier(estimators=[('RF',clf1),('KNN',clf2),('SVC',clf3)],
                        voting='hard',
                        n_jobs=-1)
# 建立函式 kfold_cross_validation() 執行 k 折交叉驗證，並回傳準確度的平均值
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.model_selection import KFold, cross_val_score
def kfold_cross_validation(scalar, model):
    """ 函式描述：執行 k 折交叉驗證
    參數：
        scalar (StandardScaler):標準化適配的結果
        model: 機器學習模型
    回傳：
        k 折交叉驗證的準確度(accuracy)平均值
    """
    # 建立管線，用來進行(標準化 -> 機器學習模型)
    make_pipeline(scalar,model)  #TODO
    pipeline = Pipeline(steps=[('StandardScalar', scalar)
                               ,('model',model)])
    # 產生 k 折交叉驗證，除以下參數設定外，其餘為預設值
    # #########################################################################
    # n_splits=5, shuffle=True, random_state=seed
    # #########################################################################
    kf =  KFold(n_splits=5,shuffle=True,random_state=seed) #TODO
    
    # 執行 k 折交叉驗證
    # #########################################################################
    # pipeline, X_train, y_train, cv=kf, scoring='accuracy', n_jobs=-1
    # #########################################################################
    cv_result = cross_val_score(pipeline,X_train_std, y_train, 
                                cv=kf, scoring='accuracy', n_jobs=-1) #TODO
    
    return cv_result.mean()  #TODO
# 利用 kfold_cross_validation()，分別讓分類器執行 k 折交叉驗證，計算準確度(accuracy)
best_acc = 0
models = [clf1,clf2,clf3,vote]
#TODO
for model in models:
    scalar = StandardScaler()
    ac_score = kfold_cross_validation(scalar, model)
    if ac_score > best_acc:
        best_acc = ac_score
# #############################################################################
print('k折交叉驗證後的最大分類準確度平均值%.4f'%round(best_acc,4))
    
# 利用訓練集的標準化結果，針對測試集進行標準化
# TODO

std = StandardScaler()
X_test_std = std.fit(X_train_score).transform(X_test)
print(X_test_std)
# 上述分類器針對測試集進行預測，並計算分類錯誤的個數與準確度
from sklearn.metrics import accuracy_score
# TODO
for model in models:
    model.fit(X_train_std,y_train)
    y_pred = model.predict(X_test_std)
    print('分類錯誤個數 = ',(y_test!=y_pred).sum())
    print('四個分類器對測試集的最大分類準確度 = %.4f'%round(accuracy_score(y_test, y_pred),4))
# #############################################################################
    
# 分別利用上述分類器預測分類
print("===== 預測分類 ======")
# TODO
new_data =[[100,70]]
data_df = pd.DataFrame(new_data)
data_std = std.fit(X_train_score).transform(data_df)
print('new data predict result = ',vote.predict(data_std))
