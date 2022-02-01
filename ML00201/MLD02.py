from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

#  載入scikit-learn資料集範例資料
# TODO
X,y = make_blobs(n_samples=200,centers=4,random_state=0,cluster_std=0.5)
plt.scatter(X[:, 0], X[:, 1], s=50)
#inertia_集群內誤差平方和，做轉折判斷法的依據
# TODO
wcss = []
for i in range(1,16):
    kmeans = KMeans(n_clusters = i,init='k-means++',max_iter=200,random_state=0,n_init=15)
    kmeans.fit(X)
    #實作
    # TODO
    wcss.append(kmeans.inertia_)

plt.plot(range(1,16), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#from 1 to 4
for i in range(1,5):
    kmeans = KMeans(n_clusters = i,init='k-means++',max_iter=200,random_state=0,n_init=15)  
    kmeans.fit(X)
    result = kmeans.inertia_
    print('result:',result)
    
# TODO
kmeans_fit = KMeans(n_clusters = 4,init='k-means++',max_iter=200,random_state=0,n_init=15)  
kmeans_fit.fit(X)
centers = kmeans_fit.cluster_centers_
kmeans_predict=kmeans_fit.predict(X)
print("cluster_centers=",centers)



