from __future__ import annotations

import datetime
import json

import numpy as np
from rich import print
import errno
import math
import time
from copy import deepcopy
from typing import Callable, NewType, TypeVar
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import tensorflow as tf
from sklearn.utils.extmath import row_norms

from src.clustering.kmeans import Kmeans, KMeansTF26

Point = NewType('Point', list[float])
Points = NewType('Points', list[Point])

'''
Type Generics
'''
Args = TypeVar('Args')
Res = TypeVar('Res')

Func = Callable[[Args], Res]
ArgsFactory = Callable[[int], Args]


def exponential(xs, y0):
    '''
    y = a * e ** x

    :param xs:
    :param y0:
    :return:
    '''
    a = y0 / (math.e ** (xs[0]))
    return [a * math.e ** (x) for x in xs]


def logrithmic(xs, y0):
    '''
    y = a * log(x)

    :param xs:
    :param y0:
    :return:
    '''
    a = y0 / (math.log(xs[0]))
    return [a * math.log(x) for x in xs]


def linearithmic(xs, y0):
    '''
    y = a * x * log(x)

    :param xs:
    :param y0:
    :return:
    '''
    a = y0 / (xs[0] * math.log(xs[0]))
    return [a * x * math.log(x) for x in xs]


def polynomial(xs, y0, n):
    '''
    y = log( y0 / xs[0]  * x)
    :param xs:
    :param y0:
        anchor for translate line at the same start
        a * xs[0] ** n = y0
        a =
    :param log:
    :return:
    '''
    a = y0 / (xs[0] ** n)
    return [a * x ** n for x in xs]


def plot_lines(xs, results, names, loglog=True):
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(15, 6), dpi=80)

    # ax1.set_titile('Common running time')
    if loglog:

        xs_log = [math.log(x) for x in xs]
        for res, name in zip(results, names):
            ax0.plot(xs_log, [math.log(y + 0.000000001) for y in res], label=name)
        ax0.legend()
        # Comparison
        polys = [f'Power of {i}' for i in range(0, 4)]

        yss = [polynomial(xs, xs[0], i) for i in range(0, 4)]
        yss.append(logrithmic(xs, xs[0]))
        yss.append(linearithmic(xs, xs[0]))

        for res, name in zip(yss, polys):
            ax1.plot(xs_log, [math.log(y) for y in res], label = name)
        ax1.legend()
        plt.show()
    else:
        xs_log = [x for x in xs]
        for res, name in zip(results, names):
            ax0.plot(xs_log, [y for y in res], label=name)
        ax0.legend()

        # Comparison
        yss = [polynomial(xs, xs[0], i) for i in range(0, 4)]
        polys = [f'Power of {i}' for i in range(0, 4)]
        # yss.append(logrithmic(xs, xs[0]))
        # yss.append(linearithmic(xs, xs[0]))
        for res, name in zip(yss, polys):
            ax1.plot(xs_log, [y for y in res], label=name)
        ax1.legend()
        plt.show()

    pass


class DoublingTest:
    def __init__(self,
                 name: str,
                 func: Func,
                 args: iter[int],
                 factory: ArgsFactory = None
                 ) -> None:
        '''
        Doubling test for the function func

        :param func:
        :param args: integer array which doubles its value
        :param factory:
        '''
        self.name = name
        self.func = func
        self.args = args
        self.factory = factory
        self.ys = []

    def testing(self) -> None:
        # print(self.func)
        # print(self.iter_args)
        # for arg in self.iter_args:
        #     print(arg)
        #     raise errno
        self.ys = []
        N = len(self.args)
        for i, (arg, size) in enumerate(zip(self.iter_args, self.args)):
            print(f'Beginning testing {i}/{N} of {self.func} on size of 2 ** {int(math.log(size)):3}. Time: ', end='')
            start = time.time()
            self.func(arg)
            self.ys.append(time.time() - start)
            print(str(datetime.timedelta(seconds=self.ys[-1])))


        return self.ys

    def show(self) -> None:
        if len(self.ys) == 0: self.testing()
        xs = list(self.args)
        plt.plot(xs, self.ys)
        plt.title = self.name
        plt.show()

    @property
    def iter_args(self):
        return map(self.factory, deepcopy(self.args)) if self.factory else deepcopy(self.args)


class RTComparisions:
    def __init__(self,
                 names: list[str],
                 funcs: list[Func],
                 args: iter[int],
                 factory: ArgsFactory = None
                 ) -> None:
        self.names = names
        self.funcs = funcs
        self.args = args
        self.factory = factory

    def show(self):
        tests = [DoublingTest(name, f, self.args, self.factory) for name, f in zip(self.names, self.funcs)]
        xs = list(self.args)
        results = [test.testing() for test in tests]

        data = {'names': self.names, 'xs':xs, 'results':results}
        self.save(data)
        plot_lines(xs, results, self.names)
        pass
    def save(self, data):
        experiment = {
            'time': datetime.datetime.now().__str__(),
            'data': data,
        }
        file = f'{time.time()}.json'
        with open(file, 'w') as f:
            json.dump(experiment, f)




def demo_DT():
    square = lambda x: x ** 2
    dtest = DoublingTest('print', print, iter(range(8)), lambda x: x ** 2)
    dtest.testing()
    dtest.testing()
    dtest1 = DoublingTest('print', print, map(lambda x: -x, range(5)))
    dtest1.testing()

    sleep = lambda x: time.sleep(x / 10)
    dtest2 = DoublingTest(sleep, range(1, 10))
    dtest2.show()

    print(list(map(square, iter(range(5)))))


def demoRT():
    func1 = lambda x: time.sleep(x / 100)
    func2 = lambda x: time.sleep(x ** 2 / 100)
    func3 = lambda x: time.sleep(x ** 3 / 100)

    funcs = []


def draw_polynomial():
    n = 10
    xs = [2 ** i for i in range(2, n)]
    yss = [polynomial(xs, xs[0], i) for i in range(0, 6)]
    yss.append(logrithmic(xs, xs[0]))
    yss.append(linearithmic(xs, xs[0]))
    # print(xs)
    # print(yss)
    # plot_lines(xs, yss)


class Make_Clusters:
    def __init__(self, n, dim=2):
        self.n = n
        self.dim = dim

    def __call__(self, size):
        return make_blobs(n_samples=size, n_features=self.dim, centers=self.n)[0]


def kmeansRunTimeTesting(n_centroids, dim=2):
    make_clusters = Make_Clusters(n_centroids, dim)
    kmeans = KMeans(n_clusters=n_centroids).fit
    kmeans1 = Kmeans(n_clusters=n_centroids).fit
    kmeans2 = KMeansTF26(n_clusters=n_centroids).fit
    kmeans_testing = RTComparisions(
        ['scikit', 'cpu', 'gpu'],
        [kmeans, kmeans1, kmeans2],
        [2 ** i for i in range(10, 25)],
        make_clusters
    )
    kmeans_testing.show()


def kmeansTesting(n_centroids, dim=2):
    points = Make_Clusters(n_centroids, dim)(100)
    xs, ys = zip(*points)
    plt.scatter(xs, ys)
    s = 1
    kmeans = KMeans(n_clusters=n_centroids).fit(points)
    # kmeans1 = Kmeans(n_clusters=n_centroids).fit(points)

    kmeans3 = KMeansTF26(n_clusters=n_centroids).fit(tf.constant(points))

    for x, y in kmeans.cluster_centers_:
        plt.scatter(x, y, c='red', marker='*', s=22 ** 2)
    for x, y in kmeans3.centroids:
        plt.scatter(x, y, c='black', marker='^', s=22 ** 1.5)
    plt.show()

def show_plot(file = '1643696629.883552.json', loglog = False):

    with open(file, 'r') as f:
        experiment = json.load(f)

    record = experiment['data']
    data = record['xs'], record['results'], record['names']
    plot_lines(*data,loglog=loglog)

if __name__ == '__main__':
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # draw_polynomial()
    # kmeansRunTimeTesting(3)
    show_plot()

    # kmeansTesting(2)
    # demoRT()
    # demo_DT()
    # print(isinstance(lambda x : x, ))

    # physical_devices = tf.config.list_physical_devices('GPU')
    # print("Num GPUs:", len(physical_devices))
    #
    # mat = np.array([[1.0,2],[3,4], [5,6]])
    # print(mat)
    # print(mat.mean(axis=0))
    # print(mat - mat.mean(axis=0))
    #
    # mat_tf = tf.constant(mat)
    # mean_tf = tf.reduce_mean(mat, axis=0)
    # print(tf.reduce_mean(mat, axis=0))
    # print(mat_tf - mean_tf)

    # mat -= mat.mean(axis=0)
    # print(mat)
    # norm = row_norms(mat, squared=True)
    # print(norm)
    #
    # mask = np.random.randint(3, size=10)
    # print(mask == 0)
