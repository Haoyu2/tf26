import random
import time

import tensorflow as tf

points = tf.random.uniform((3, 2))
# print(points)
# sb =tf.subtract(points, tf.constant([0.1,0.1]))
# print(tf.subtract(points, tf.constant([0.1,0.1])))
# ssb = tf.square(sb, 2)
# print(ssb)
# ssb_sum = tf.reduce_sum(ssb, 1)
# print(ssb_sum)
# ssb_sum_sqrt  =tf.sqrt(ssb_sum)
# print(ssb_sum_sqrt)
# print(tf.reduce_sum(ssb_sum_sqrt))
#
# points = tf.Variable(points)
#
# x = tf.Variable(3.0)

def distance_sum(cent):
    return tf.reduce_sum( tf.sqrt(tf.reduce_sum(tf.square( tf.subtract(points, cent), 2), 1)))

def d_dis(cent, points = points):
    sb = tf.subtract(points, cent)
    dis = tf.sqrt(tf.reduce_sum(tf.square(sb, 2), 1))
    # print(sb)
    # print(tf.reshape(dis, (-1, 1)))
    # print(sb/ tf.transpose(dis))
    return tf.reduce_sum(sb / tf.reshape(dis, (-1, 1)) )


# print(distance_sum(tf.constant([0.1, 0.1])))

# cent = tf.Variable([0.1, 0.1])
# print(d_dis(cent))
# with tf.GradientTape() as tape:
#     y = distance_sum(cent)
#
# dd = tape.gradient(y, x)
# print(dd)


def centroid(points):
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

# points = [[random.random(), random.random()] for i in range(10000)]
# import matplotlib.pyplot as plt
#
# points = tf.constant(points)
#
# xs, ys = zip(*points)
# plt.scatter(xs, ys)
# start = time.time()
# x0, y0 = centroid(points)
# print(time.time() - start)
# print(x0,y0)
# plt.scatter(x0, y0, c='red')
# plt.show()