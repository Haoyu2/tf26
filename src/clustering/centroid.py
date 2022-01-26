import math
import random
import time

import matplotlib.pyplot as plt
import tensorflow as tf
def getCentroid2D(points: list[list[float]])->list[float]:
    '''

    :param points:
    :return:
    '''
    eps = 1e-7
    alpha = 1.0
    decay = 1e-3
    N = len(points)

    xs, ys = zip(*points)
    x0, y0 = math.fsum(xs) / N, math.fsum(ys) / N
    dis = lambda x0, y0 : sum((x0 - x) ** 2 + (y - y0) ** 2 for x, y in points)
    ddis = lambda x0, y0, x, y: math.sqrt((x - x0)**2 + (y-y0)**2)
    ddis_x = lambda x0, x, dd: (x0 - x) / dd
    ddis_y = lambda y0, y, dd: (y0 - y) / dd

    batch_size = len(points)
    while True:
        random.shuffle(points)
        prex, prey = x0, y0
        for i in range(0, N, batch_size):
            dx = dy = 0.0
            for j in range(i, min(i+batch_size, N)):
                x, y = points[j]
                dd = ddis(x0, y0, x, y)
                dx += (x0 - x) / dd
                dy += (y0 - y) / dd
            x0 -= alpha * dx
            y0 -= alpha * dy

            alpha *= (1.0 - decay)

        if ((x - prex) ** 2 + (y - prey) ** 2 ) ** 0.5 < eps:
            return x, y

def getMinDistSum(positions) -> float:
    eps = 1e-7
    alpha = 1.0
    decay = 1e-3

    n = len(positions)
    # 调整批大小
    batchSize = n

    x = sum(pos[0] for pos in positions) / n
    y = sum(pos[1] for pos in positions) / n

    # 计算服务中心 (xc, yc) 到客户的欧几里得距离之和
    getDist = lambda xc, yc: sum(((x - xc) ** 2 + (y - yc) ** 2) ** 0.5 for x, y in positions)

    while True:
        # 将数据随机打乱
        random.shuffle(positions)
        xPrev, yPrev = x, y

        for i in range(0, n, batchSize):
            j = min(i + batchSize, n)
            dx, dy = 0.0, 0.0

            # 计算导数，注意处理分母为零的情况
            for k in range(i, j):
                pos = positions[k]
                dx += (x - pos[0]) / (math.sqrt((x - pos[0]) * (x - pos[0]) + (y - pos[1]) * (y - pos[1])) + eps)
                dy += (y - pos[1]) / (math.sqrt((x - pos[0]) * (x - pos[0]) + (y - pos[1]) * (y - pos[1])) + eps)

            x -= alpha * dx
            y -= alpha * dy

            # 每一轮迭代后，将学习率进行衰减
            alpha *= (1.0 - decay)

        # 判断是否结束迭代
        if ((x - xPrev) ** 2 + (y - yPrev) ** 2) ** 0.5 < eps:
            return x, y


    return getDist(x, y)







if __name__ == '__main__':
    points = [[random.random(), random.random()] for i in range(10000)]
    # print(points)
    # print(getCent
    # roid2D(points))
    xs, ys = zip(*points)
    plt.scatter(xs, ys)
    # x0, y0 = getMinDistSum(points)

    points_tensor = tf.constant(points)
    # print(points_tensor)

    ps = tf.constant([[1,2], [3, 4], [5, 6]])
    print(ps[:2])
    print(tf.reduce_mean(ps, 0))

    start = time.time()
    x0, y0 = getMinDistSum(points)
    print(time.time() - start)
    # print(getMinDistSum(points))
    print(x0,y0)
    plt.scatter(x0, y0, c='red')
    plt.show()