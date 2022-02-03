import numpy as np
# TODO
from sklearn.cluster import DBSCAN
from sklearn import metrics
input_file = ('data_perf.txt')
# Load data 載入資料
# TODO
X = []
with open(input_file,'r') as txtf:
    for line in txtf.readlines():
        data = [float(i) for i in line.split(',')]
        X.append(data)
X = np.array(X)    
print(X)
# Find the best epsilon 
eps_grid = np.linspace(0.3,1.2,num=10)
silhouette_scores = []
eps_best = eps_grid[0]
silhouette_scores_max = -1
# TODO
for eps in eps_grid:
    # Train DBSCAN clustering model 訓練DBSCAN分群模型
    # ################
    model = DBSCAN(eps = eps, min_samples=5).fit(X)
    # min_samples = 5
    # ################
    

    # Extract labels 提取標籤
    labels = model.labels_

    # Extract performance metric 提取性能指標
    silhouette_score = round(metrics.silhouette_score(X, labels),4)
    silhouette_scores.append(silhouette_score)

    print("Epsilon:", eps, " --> silhouette score:", silhouette_score)

    # TODO
    if silhouette_score > silhouette_scores_max:
        silhouette_scores_max = silhouette_score
        eps_best = eps
        model_best = model
        labels_best = labels

# Best params
print("Best epsilon =",eps_best           )
print("Best Silhouette = ",silhouette_scores_max) 
# Associated model and labels for best epsilon
model = model_best   # TODO
labels = labels_best  # TODO

# Check for unassigned datapoints in the labels
# TODO
offset = 0
if -1 in labels:
    offset = 1

# Number of clusters in the data 
# TODO
print("Estimated number of clusters =",len(set(labels)) - offset)

# Extracts the core samples from the trained model
# TODO
X = []
with open('data_perf_add.txt','r') as txtf:
    for line in txtf.readlines():
        data = [float(i) for i in line.split(',')]
        X.append(data)
X = np.array(X)    
# Find the best epsilon 
eps_grid = np.linspace(0.3,1.2,num=10)
silhouette_scores = []
eps_best = eps_grid[0]
silhouette_scores_max = -1
# TODO
for eps in eps_grid:
    # Train DBSCAN clustering model 訓練DBSCAN分群模型
    # ################
    model = DBSCAN(eps = eps, min_samples=5).fit(X)
    # min_samples = 5
    # ################
    

    # Extract labels 提取標籤
    labels = model.labels_

    # Extract performance metric 提取性能指標
    silhouette_score = round(metrics.silhouette_score(X, labels),4)
    silhouette_scores.append(silhouette_score)

    print("Epsilon:", eps, " --> silhouette score:", silhouette_score)

    # TODO
    if silhouette_score > silhouette_scores_max:
        silhouette_scores_max = silhouette_score
        eps_best = eps
        model_best = model
        labels_best = labels

# Best params
print("Best epsilon =",eps_best)
print("Best Silhouette = ",silhouette_scores_max) 
