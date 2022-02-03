from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

#  載入scikit-learn資料集範例資料
# TODO
X,y = make_blobs(n_samples=200,centers=4,cluster_std=0.5,random_state=0)
plt.scatter(X[:,0], X[:,1],s=50)

#inertia_集群內誤差平方和，做轉折判斷法的依據
# TODO
wcss = []
for i in range(1,16):
    #實作
    # TODO
    kmeans = KMeans(n_clusters=i,n_init=15,random_state=0,max_iter=200)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,16), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

count_cluster = 0
for w in wcss:
    if w >= 90:
        count_cluster += 1
print('集群內誤差平方和大於90以上,可以分為',count_cluster,'群')
kmeans = KMeans(n_clusters=count_cluster,n_init=15,random_state=0,max_iter=200) 
# TODO
kmeans_fit = kmeans.fit(X)
kmeans_predict=kmeans_fit.predict(X)
print("cluster_centers=",kmeans_fit.cluster_centers_)
print('分群後最小中心點X的位置',min(kmeans_fit.cluster_centers_[:,0]))
print('分群後最大中心點Y的位置',max(kmeans_fit.cluster_centers_[:,1]))