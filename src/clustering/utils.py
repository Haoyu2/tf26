from sklearn.datasets import make_blobs
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gradients import centroid
def make_clusters(n: int, num: int = 3000) -> list:
    cents = np.random.randint(0, num, (n, 2))
    points, label, cents = make_blobs(n_samples=num,
                                      centers=cents,
                               n_features=2,
                               cluster_std=num**0.5,
                                return_centers=True
                               )
    return cents, points, label

def _scratch():
    cents, points, label = make_clusters(3, 1000)
    xs, ys = zip(*points)
    zero = points[label == 0]
    # print(cents)
    import matplotlib.pyplot as plt
    plt.scatter(xs, ys, c = label)
    plt.show()


def dis(points, center):
    return tf.reduce_sum(tf.square(tf.subtract(points, center), 2),1)

def kmeans(points, n):
    label = np.random.randint(0, n, (len(points)))
    print(label == 0)

    # cents = np.random.randint(0, 1000, (n, 2))
    xs, ys = zip(*points)
    plt.scatter(xs, ys, c =label)

    centers = [centroid(tf.boolean_mask(points, label == i)) for i in range(n)]
    cx, cy = zip(*centers)
    plt.scatter(cx, cy, c = 'red')

    plt.show()
    # print(centers)
    for i in range(1):
        label = []
        for center in centers:
            print(center)
            # label.append(dis(points, center))
            # diss = tf.stack(label, axis=1)
            # print(tf.argmin(diss, axis=1))

    #
    # print('aa',cents)
    # label = []
    # for cent in cents:
    #     label.append(dis(points, cent))
    # print(label)
    # print(tf.stack(label, axis=1))
    # print(tf.argmin(tf.stack(label,axis=1), axis=1))
    #
    # print(tf.reshape(tf.concat(label, 0), (3, -1)))

def _scratch1():
    n = 3
    cents, points, label = make_clusters(n, 10)
    kmeans(points, n)




if __name__ == '__main__':
    _scratch1()



