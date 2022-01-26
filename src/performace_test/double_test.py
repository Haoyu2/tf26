from __future__ import annotations

import datetime

from rich import print
import errno
import math
import time
from copy import deepcopy
from typing import Callable, NewType, TypeVar
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

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


def plot_lines(xs, results, loglog=True):
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex=True)
    # ax1.set_titile('Common running time')
    if loglog:

        xs_log = [math.log(x) for x in xs]
        for res in results:
            ax0.plot(xs_log, [math.log(y) for y in res])

        # Comparison
        yss = [polynomial(xs, xs[0], i) for i in range(0, 6)]
        yss.append(logrithmic(xs, xs[0]))
        yss.append(linearithmic(xs, xs[0]))
        for res in yss:
            ax1.plot(xs_log, [math.log(y) for y in res])
        plt.show()
    else:
        for res in results:
            plt.plot(xs, res)
        plt.show()

    pass


class DoublingTest:
    def __init__(self,
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
        for i,  (arg, size) in enumerate(zip(self.iter_args, self.args)):
            print(f'Beginning testing {i}/{N} of {self.func} on size of 2 ** {int(math.log(size)):3}. Time: ', end='')
            start = time.time()
            self.func(arg)
            self.ys.append(time.time() - start)
            print(str(datetime.timedelta(seconds=self.ys[-1])) )
        return self.ys

    def show(self) -> None:
        if len(self.ys) == 0: self.testing()
        xs = list(self.args)
        plt.plot(xs, self.ys)
        plt.show()

    @property
    def iter_args(self):
        return map(self.factory, deepcopy(self.args)) if self.factory else deepcopy(self.args)


class RTComparisions:
    def __init__(self,
                 funcs: list[Func],
                 args: iter[int],
                 factory: ArgsFactory = None
                 ) -> None:
        self.funcs = funcs
        self.args = args
        self.factory = factory

    def show(self):
        tests = [DoublingTest(f, self.args, self.factory) for f in self.funcs]
        xs = list(self.args)
        results = [test.testing() for test in tests]
        plot_lines(xs, results)
        pass



def demo_DT():
    square = lambda x: x ** 2
    dtest = DoublingTest(print, iter(range(8)), lambda x: x ** 2)
    dtest.testing()
    dtest.testing()
    dtest1 = DoublingTest(print, map(lambda x: -x, range(5)))
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
    def __init__(self, n, dim = 2):
        self.n = n
        self.dim = dim
    def __call__(self, size):
        return make_blobs(n_samples=size, n_features=self.dim, centers=self.n)[0]

def kmeansRunTimeTesting(n_centroids, dim=2):
    make_clusters = Make_Clusters(n_centroids, dim)
    kmeans = KMeans(n_clusters=n_centroids).fit
    kmeans_testing = RTComparisions(
        [kmeans],
        [2**i  for i in range(10, 30)],
        make_clusters
    )
    kmeans_testing.show()



if __name__ == '__main__':
    # draw_polynomial()
    kmeansRunTimeTesting(2)
    # demoRT()
    # demo_DT()
    # print(isinstance(lambda x : x, ))
