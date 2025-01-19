import numpy as np

def MatrixAdd(A, B):
    """
    :param A: list[list[int]]
    :param B: list[list[int]]
    :return: list[list[int]]
    """
    return np.array(A) + np.array(B)

def MatrixMul(A, B):
    """
    :param A: list[list[int]]
    :param B: list[list[int]]
    :return: list[list[int]]
    """
    return np.dot(A, B)

# 测试
A = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

B = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

print("Matrix Add:")
result_add = MatrixAdd(A, B)
print(result_add)

print("Matrix Mul:")
result_mul = MatrixMul(A, B)
print(result_mul)
