def MatrixAdd(A, B):
    """
    :param A: list[list[int]]
    :param B: list[list[int]]
    :return: list[list[int]]
    """
    n = len(A)
    result = [[0] * n for _ in range(n)]  # 创建一个与 A、B 大小相同的零矩阵
    for i in range(n):
        for j in range(n):
            result[i][j] = A[i][j] + B[i][j]  # 对应位置元素相加
    return result

def MatrixMul(A, B):
    """
    :param A: list[list[int]]
    :param B: list[list[int]]
    :return: list[list[int]]
    """
    n = len(A)
    result = [[0] * n for _ in range(n)]   # 创建一个与 A、B 大小相同的零矩阵
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]  # 矩阵乘法的公式，逐个元素相乘再相加
    return result

# 测试
A = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

B = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

print("Matrix Add:")
result_addition = MatrixAdd(A, B)
for row in result_addition:
    print(row)

print("Matrix Mul:")
result_multiplication = MatrixMul(A, B)
for row in result_multiplication:
    print(row)
