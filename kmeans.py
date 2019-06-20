#K-Means Algorithm


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset

dataset = pd.read_csv('Mall_Customers.csv')

#Goal is to segment customers based on annual income and spending score features

X = dataset.iloc[:,[3,4]].values

#We don't know how many clusters to choose

#Using elbow method to find optimal number of cluster

from sklearn.cluster import KMeans
wcss = []
for i in range (1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10) 
    #n_cluster is amount of centroids which is i, so we can compare from 1-10
    #init controls method of initialzing centroids (K-means++) chooses optimal
    #max_init is how max times the algorthm runs before converging
    #n_init is how many times it runs with same amount of clusters to ensure good answer
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
"""
plt.plot(range(1, 11), wcss) #plotting clusters vs error
plt.title('Clusters vs. Error(Elbow Method)')
plt.xlabel('Clusters')
plt.ylabel('WCSS')
"""
#Applying K-Means to dataset with optimal clusters
kmeansoptimal = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10)
y_kmeans = kmeansoptimal.fit_predict(X) #fits and predicts clusters to dataset 
#y_kmeans contains what cluter # each example fits to

#Visualizing the clusters by plotting for each y_kmeans value (represents which cluster) and using first two rows of X as x and y
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, color = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, color = 'black', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, color = 'blue', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, color = 'yellow', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, color = 'green', label = 'Cluster 5')
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, color = 'brown', label = 'Cluster Centers')
plt.legend()
plt.show()