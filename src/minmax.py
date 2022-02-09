import tensorflow as tf
import numpy as np

from src.performace_test.double_test import RTComparisions


def min_max_multiply_gpu(mat1: tf.Tensor, mat2: tf.Tensor):
    M, N = len(mat1), len(mat2[:, 0])
    res = [[0] * N for i in range(M)]
    for i in range(M):
        for j in range(N):
            res[i][j] = tf.reduce_min(tf.maximum(mat1[i], mat2[:, j])).numpy()
    return np.array(res)


def min_max_multiply(mat1, mat2):
    M, N = len(mat1), len(mat2[:, 0])
    res = [[0] * N for i in range(M)]
    for i in range(M):
        for j in range(N):
            res[i][j] = np.min(np.maximum(mat1[i,:], mat2[:, j]))
    return np.array(res)

mat = np.random.randint(100,size=(5, 3))
print(mat[0], mat[:,0])


def making_matrics(M, N=100):
    return np.random.randint(M * N, size=(M, N)), np.random.randint(M * N, size=(N, M))


def min_max_testing(dim=2):

    min_max_testing = RTComparisions(
        [ 'cpu', 'gpu'],
        [min_max_multiply, min_max_multiply_gpu],
        [2 ** i for i in range(10, 15)],
        making_matrics
    )
    min_max_testing.show()
min_max_testing()

# mat1 = np.array([[1,2], [5,4]])
# mat2 = np.array([[3,4], [7,1]])
#
# print(min_max_multiply(mat1, mat2))
# print(min_max_multiply_gpu(mat1, mat2))


#
# print(mat1[0], mat2[:,0])
# print(np.maximum(mat1[0], mat2[:,0]))
# print(np.min(np.maximum(mat1[0], mat2[:,0])))
# print(np.amin(mat1[1]))
#
