import csv
import numpy as np
import matplotlib.pyplot as plt
import random

# 1. 数据清洗和数据可视化
ages = []
salaries = []
purchased = []

# 读取数据集
with open("data.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # 跳过标题行
    for row in reader:
        age = float(row[0])
        salary = float(row[1])
        purchase = int(row[2])
        ages.append(age)
        salaries.append(salary)
        purchased.append(purchase)

# 2. 数据预处理
# 特征缩放
ages_scaled = (ages - np.mean(ages)) / np.std(ages)
salaries_scaled = (salaries - np.mean(salaries)) / np.std(salaries)

# 合并特征
X = np.column_stack((ages_scaled, salaries_scaled))
y = np.array(purchased)

# 划分训练集和测试集
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        random.seed(random_state)
    indices = list(range(len(X)))
    random.shuffle(indices)
    split_idx = int(len(X) * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 建立感知机模型
class Perceptron:
    def __init__(self, input_size, learning_rate=0.001, max_iters=10000):  # 减小学习率，增加迭代次数
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.loss = []

        # 初始化权重和偏置
        self.weights = np.random.rand(self.input_size + 1) * 2 - 1  # 权重范围在 [-1, 1]

    def forward(self, X):
        # 前向传播
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def backward(self, X, y, y_pred):
        # 反向传播
        error = y - y_pred
        self.weights[1:] += self.learning_rate * np.dot(X.T, error)
        self.weights[0] += self.learning_rate * np.sum(error)

    def fit(self, X, y):
        for i in range(self.max_iters):
            # 前向传播
            y_pred = self.forward(X)

            # 计算损失
            loss = np.mean(np.square(y - y_pred))
            self.loss.append(loss)

            # 反向传播并更新权重和偏置
            self.backward(X, y, y_pred)

            # 每100次迭代打印一次损失
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss}")

    def predict(self, X):
        # 预测
        return np.where(self.forward(X) >= 0, 1, 0)

    def decision_boundary(self, x):
        # 分类线方程: w1*x1 + w2*x2 + b = 0
        return (-self.weights[0] - self.weights[1] * x) / self.weights[2]

# 建立感知机模型
perceptron_model = Perceptron(input_size=2, learning_rate=0.001, max_iters=10000)  # 学习率过大会导致无法收敛，增加迭代次数

# 训练模型
perceptron_model.fit(X_train, y_train)

# 绘制数据可视化图和分类线
plt.figure(figsize=(10, 6))
plt.scatter(ages_scaled, salaries_scaled, c=purchased, cmap='coolwarm', alpha=0.6)
plt.xlabel('Age (Normalized)')
plt.ylabel('Estimated Salary (Normalized)')
plt.title('Scatter Plot of Age vs Estimated Salary')

# 绘制分类线
x_values = np.linspace(np.min(ages_scaled), np.max(ages_scaled), 100)
y_values = perceptron_model.decision_boundary(x_values)
plt.plot(x_values, y_values, label='Decision Boundary', color='black')
plt.legend()

plt.show()

# 绘制损失曲线图
plt.figure(figsize=(10, 6))
plt.plot(range(len(perceptron_model.loss)), perceptron_model.loss)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()

# 预测
train_predictions = perceptron_model.predict(X_train)
test_predictions = perceptron_model.predict(X_test)

# 计算准确率
train_accuracy = np.mean(train_predictions == y_train)
test_accuracy = np.mean(test_predictions == y_test)

print("训练集准确率:", train_accuracy)
print("测试集准确率:", test_accuracy)
