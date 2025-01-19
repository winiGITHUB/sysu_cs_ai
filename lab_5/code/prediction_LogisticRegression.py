import csv
import numpy as np
import matplotlib.pyplot as plt

# 读取数据集
age = []
salary = []
purchased = []
with open("data.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # 跳过标题行
    for row in reader:
        age.append(float(row[0]))
        salary.append(float(row[1]))
        purchased.append(int(row[2]))

# 数据归一化处理
age_normalized = (age - np.mean(age)) / np.std(age)
salary_normalized = (salary - np.mean(salary)) / np.std(salary)

# Logistic Regression
class LogisticRegression:
    def __init__(self, lr=0.1, num_iterations=10000):
        self.lr = lr
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.loss = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # 计算损失并保存
            loss = - (1 / n_samples) * np.sum(y * np.log(y_predicted + 1e-8) + (1 - y) * np.log(1 - y_predicted + 1e-8))
            self.loss.append(loss)

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

# 训练模型
X = np.column_stack((np.ones(len(age_normalized)), age_normalized, salary_normalized))
y = np.array(purchased)

# 尝试不同的学习率
learning_rates = [0.001, 0.01, 0.1, 0.5, 1]

# 记录每个学习率下的准确率
accuracies = []

for lr in learning_rates:
    # 创建并训练模型
    model = LogisticRegression(lr=lr, num_iterations=10000)
    model.fit(X, y)
    
    # 计算预测准确率
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    accuracies.append(accuracy)
    print(f"Learning Rate: {lr}, Accuracy: {accuracy}")

    # 数据可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(age_normalized, salary_normalized, c=purchased, cmap='bwr', label='Purchased')
    plt.xlabel('Age (Normalized)')
    plt.ylabel('Estimated Salary (Normalized)')
    plt.title(f'Visualization of Data (Normalized), Learning Rate: {lr}')

    # 绘制分类线
    x_values = np.array([np.min(age_normalized), np.max(age_normalized)])
    y_values = -(model.weights[1] * x_values + model.bias) / model.weights[2]
    plt.plot(x_values, y_values, label='Decision Boundary', color='black')
    plt.legend()

    plt.show()

    # 损失曲线图
    plt.plot(range(len(model.loss)), model.loss)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve, Learning Rate: {lr}')
    plt.show()

# 可视化学习率对准确率的影响
plt.figure(figsize=(8, 5))
plt.plot(learning_rates, accuracies, marker='o')
plt.xscale('log')  # 对学习率取对数以便观察
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('Effect of Learning Rate on Accuracy')
plt.grid(True)
plt.show()
