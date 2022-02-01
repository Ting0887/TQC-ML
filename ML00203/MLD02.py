import pandas as pd
# 載入寶可夢資料
# TODO
df = pd.read_json('pokemon.json')
# 取出目標欄位
# TODO
X_train = df.iloc[:,:-1]
X_train.head()
# 特徵標準化
from sklearn.preprocessing import StandardScaler
# TODO
scalar = StandardScaler()
scalar.fit(X_train)
X_train_std = scalar.transform(X_train)
print(X_train_std)
# 利用 Hierarchical Clustering 進行分群，除以下參數設定外，其餘為預設值
# #############################################################################
# n_clusters=4, affinity='euclidean', linkage='ward'
# #############################################################################
from sklearn.cluster import AgglomerativeClustering
# TODO
model = AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='ward')
model.fit(X_train_std)
# 計算每一群的個數
# TODO
nums_lst = model.labels_.tolist()
for i in range(len(set(nums_lst))):
    print('Cluster',i+1,'個數',nums_lst.count(i))

# 找到 Speed 有遺漏值的兩隻寶可夢，並填入組內平均
# TODO
df['cluster'] = model.labels_
for i in df.index:
    if pd.isna(df.iloc[i,-2]):
        print(df.iloc[i,:].tolist())
        poke = df[df['cluster']==df.iloc[i,-1]].mean()
        print('Speed = %d' %round(poke['Speed'],0))