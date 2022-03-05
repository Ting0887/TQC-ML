import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
# TODO
symbol_file = 'symbol_map.json'

# TODO
symbols = pd.read_json(symbol_file,typ="Series")
columns = ["index_count","date","open","close","name"]
df_source = pd.DataFrame()
for symbol in symbols.index:
# TODO
    csv_file = symbol + '.csv'
    df_temp = pd.read_csv(csv_file)
# The daily fluctuations of the quotes 報價的每日波動
# TODO
    df_temp['name'] = symbols.loc[symbol]
    df_temp.columns = columns
    df_source = pd.concat([df_source,df_temp])

df_source['diff'] = df_source['close'] - df_source['open']
df_source.drop(['open','close','index_count'],axis=1,inplace=True)

df_result = None
for name,group in df_source[['date','diff']].groupby(df_source['name']):
    if df_result is None:
        df_result = group.rename(columns={'diff':name})
    else:
        df_result = pd.merge(df_result,
                             group.rename(columns={"diff":name}))

# Build a graph model from the correlations 根據相關性建立圖模型
# TODO
from sklearn.covariance import GraphicalLassoCV
edge_model = GraphicalLassoCV()

# Standardize the data 標準化資料
# TODO
df_result.drop(['date'],axis=1,inplace=True)
stock_dataset = np.array(df_result).astype(np.float64)
stock_dataset /= np.std(stock_dataset,axis=0)
select_stocks = df_result.columns.tolist()
# Train the model 訓練模型
# TODO
from sklearn.cluster import affinity_propagation
edge_model.fit(stock_dataset)

# Build clustering model using affinity propagation 用相似性傳播構建分群模型
# TODO
_,labels = affinity_propagation(edge_model.covariance_)
# Print the results of clustering 列印分群結果
# TODO
n_labels = max(labels)
for i in range(n_labels+1):
    stocks = np.array(select_stocks)[labels==i].tolist()
    print("Cluster", i+1, "-->",stocks)
