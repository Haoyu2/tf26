import numpy as np
from numpy.linalg import norm
import tensorflow as tf
from sklearn.cluster import KMeans
import random

class KMeansTF26:
    def __init__(self, n_clusters, max_iter=100, random_state=123):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def initializ_centroids(self, X):

        return tf.gather(X,
                indices=np.random.randint(len(X), size=self.n_clusters))
    def compute_centroids(self, X, labels):
        centroids = []
        for k in range(self.n_clusters):
            centroids.append(tf.reduce_mean(X[labels == k], axis=0))
        return tf.stack(centroids)

    def compute_distance(self, X, centroids):
         return [tf.reduce_sum(tf.square(tf.subtract(X, cent), 2), 1) for cent in centroids]
    def find_closest_cluster(self, distances):
        return tf.argmin(distances, axis=0)
    def fit(self, X):
        X = tf.constant(X)
        self.centroids = self.initializ_centroids(X)
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(X, self.labels)
            print(old_centroids)
            print()
            print(self.centroids)
            print(old_centroids - self.centroids)
            if tf.reduce_sum(tf.abs(old_centroids - self.centroids)) < self.n_clusters :
                break
        return self



class Kmeans:
    '''Implementing Kmeans algorithm.'''

    def __init__(self, n_clusters, max_iter=100, random_state=123):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def initializ_centroids(self, X):
        np.random.RandomState(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids

    def compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids

    def compute_distance(self, X, centroids):
        distance = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)

    def compute_sse(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))

    def fit(self, X):
        self.centroids = self.initializ_centroids(X)
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(X, self.labels)
            if np.all(old_centroids == self.centroids):
                break
        self.error = self.compute_sse(X, self.labels, self.centroids)
        return self

    def predict(self, X):
        distance = self.compute_distance(X, self.centroids)
        return self.find_closest_cluster(distance)




def d_dis(cent, points):
    sb = tf.subtract(points, cent)
    dis = tf.sqrt(tf.reduce_sum(tf.square(sb, 2), 1))
    return tf.reduce_sum(sb / tf.reshape(dis, (-1, 1)) )

class KMeansTF:
    def __init__(self, n_clusters:int):
        self.n_clusters = n_clusters

    def centroid(self, points):
        cent = tf.reduce_mean(points, 0)
        eps = 1e-7
        alpha = 1.0
        decay = 1e-3
        while True:
            pre = tf.identity(cent)
            dd = d_dis(cent, points=points)
            # print(dd)
            cent += alpha * dd
            if tf.reduce_sum(tf.square(tf.subtract(cent, pre))) ** 0.5 < eps:
                return cent
            alpha *= (1.0 - decay)

    def distance(self, points, cent):
        return tf.reduce_sum(tf.square(tf.subtract(points, cent), 2), 1)

    def fit(self, X):

        X -= tf.reduce_mean(X, axis=0)
        # mask
        mask = np.random.randint(self.n_clusters, size=len(X))

        for i in range(10):
            data = [tf.boolean_mask(X, mask == i) for i in range(self.n_clusters)]
            centroids = [self.centroid(points) for points in data]
            diss = [self.distance(X, cent) for cent in centroids]
            mask = tf.argmin(tf.stack(diss),axis=0)
        self.centroids = centroids
        return self





        pass

def demo():

    kmeans = KMeansTF(3)
    points = [[random.random(), random.random()] for i in range(10)]
    points = tf.constant(points)
    print(points)
    print(kmeans.distance(points, [0,0]))

    cents = [[0,0], [0.1,0.2], [0.3,0.5]]

    diss = [kmeans.distance(points, [0.1*i,1 - 0.1*i]) for i in range(3)]
    print(tf.stack(diss))
    print(tf.argmin(tf.stack(diss), axis=0))


def demo1():

    kmeans = KMeansTF(1)
    points = [[random.random(), random.random()] for i in range(10)]
    points = tf.constant(points)

    import matplotlib.pyplot as plt

    points_tf = tf.constant(points)

    xs, ys = zip(*points)
    plt.scatter(xs, ys)
    x0, y0 = kmeans.centroid(points_tf)
    print(x0, y0)
    plt.scatter(x0, y0, c='red')
    sk_kmeans = KMeans(n_clusters=1).fit(points)
    x1, y1 = sk_kmeans.cluster_centers_[0]
    plt.scatter(x1, y1, c='black')

    x3, y3 = kmeans.fit(points_tf).centroids[0]
    plt.scatter(x3, y3)

    plt.show()


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print(tf.argmin([tf.constant([1,3,5]), tf.constant([2,2, 1])],axis=0))
