from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3,7,8,9])
y = np.array([7,6,7,3,2,2])

plt.scatter(x, y)
plt.show()

X = np.array(zip(x,y))

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)


centroids = kmeans.cluster_centers_
labels = kmeans.labels_

colours = ['g.', 'r.', 'c.', 'y.', 'm.', 'k.' ,'w.', 'go']
for i in range(len(X)):
    colour = colours[labels[i]%len(colours)]
    plt.plot(X[i][0],X[i][1], colour, '10')

# Display centroids as well
plt.scatter(centroids[:,0], centroids[:,1], marker="x", s = 150, linewidths=4)
plt.show()