 #############################################################################
# 本題參數設定，請勿更改
seed = 0    # 亂數種子數
# #############################################################################

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
# 載入寶可夢資料
# TODO
file_name = "pokemon.csv"
# 注意欄位名稱設定
columns = ['Defense','SpecialAtk']
df = pd.read_csv(file_name)

# # 取出目標欄位
X = df[columns]
y = df[['Type1']]

# 編碼 Type1
from sklearn import preprocessing
# TODO
label_encoder = preprocessing.LabelEncoder()
y['Type1'] = label_encoder.fit_transform(y['Type1'])

# 切分訓練集、測試集，除以下參數設定外，其餘為預設值
# #########################################################################
# X, y, test_size=0.2, random_state=seed
# #########################################################################
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=seed)

# 注意要複製出來一份，等下要用來標準化測試集    
X_train_source = X_train.copy(deep=False)

# 特徵標準化
from sklearn.preprocessing import StandardScaler
# TODO

stand_scalar = StandardScaler()
for name in columns:
    X_train[name] = stand_scalar.fit_transform(X_train[[name]])

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
# 注意名稱為clf1,clf2,clf3,voting
clf1 = RandomForestClassifier(n_estimators=10, random_state=seed)
clf2 = KNeighborsClassifier(n_neighbors=4)
clf3 = SVC(gamma=.1, kernel='rbf', probability=True)
voting = VotingClassifier(estimators=[('RF', clf1), ('kNN', clf2), ('SVC', clf3)], 
         voting='hard', n_jobs=-1) 

# 建立函式 kfold_cross_validation() 執行 k 折交叉驗證，並回傳準確度的平均值
# 注意要多匯入Pipeline
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
    # 注意寫法
    make_pipeline(scalar, model)
    pipeline = Pipeline(steps=[("StandardScaler",scalar),\
                               ("model",model)])  #TODO
    
    # 產生 k 折交叉驗證，除以下參數設定外，其餘為預設值
    # #########################################################################
    # n_splits=5, shuffle=True, random_state=seed
    # #########################################################################
    kf =  KFold(n_splits=5, shuffle=True, random_state=seed) #TODO
    
    # 執行 k 折交叉驗證
    # #########################################################################
    # pipeline, X_train, y_train, cv=kf, scoring='accuracy', n_jobs=-1
    # #########################################################################
    cv_result = cross_val_score(pipeline, X_train, y_train, cv=kf,\
                                scoring='accuracy', n_jobs=-1)
    return  cv_result.mean()#TODO

# 利用 kfold_cross_validation()，分別讓分類器執行 k 折交叉驗證，計算準確度(accuracy)
# 注意自訂函式    
def get_round_value(value):
    return "{:.4f}".format(round(value,4))
    
#TODO
best_acc = 0
models = [clf1,clf2,clf3,voting]
for model in models:
    stand_scalar = StandardScaler()
    ac_score = kfold_cross_validation(stand_scalar,model)
    if ac_score > best_acc:
        best_acc = ac_score
# #############################################################################
print("k折交叉驗證後的最大分類準確度平均值（四捨五入取至小數點後第四位）:",\
      get_round_value(best_acc))

    
# 利用訓練集的標準化結果，針對測試集進行標準化
# TODO
# TODO
stand_scalar = StandardScaler()
for name in columns:
    X_test[name] = stand_scalar.fit(\
                                    X_train_source[[name]]).transform(\
                                    X_test[[name]])


# 上述分類器針對測試集進行預測，並計算分類錯誤的個數與準確度
from sklearn.metrics import accuracy_score
# TODO
error_num = 0
best_acc = 0
for model in models:
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    ac_score = accuracy_score(y_test,y_pred)
    if ac_score > best_acc:
        best_acc = ac_score
        # 注意寫法
        error_num = y_pred.shape[0]- accuracy_score(y_test,y_pred,\
                                                    normalize=False)
# #############################################################################
print("四個分類器對測試集的最大分類準確度（四捨五入取至小數點後第四位）:",\
      get_round_value(best_acc))
print("四個分類器對測試集的最小分類錯誤樣本數",\
      error_num)    
# 分別利用上述分類器預測分類
print("===== 預測分類 ======")
# TODO
df_test = pd.DataFrame({'Defense':100,'SpecialAtk':70},index=[0])
stand_scalar = StandardScaler()
for name in columns:
    # 注意利用訓練集的標準化結果，針對測試集進行標準化
    df_test[name] = stand_scalar.fit(\
                                    X_train_source[[name]]).transform(\
                                    df_test[[name]])
X_test = df_test                                                        
voting.fit(X_train,y_train)
y_pred = voting.predict(X_test)
# 注意寫法
print("一個未知寶可夢的Defense=100, SpecialAtk=70，請填入投票分類器預測其Type1的分類選項",label_encoder.classes_[int(y_pred)])
