import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

# 定义一个简单的卷积神经网络（CNN）模型
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # 定义特征提取部分的网络层
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),  # 输入通道数为3（RGB图像），输出通道数为6，卷积核大小为5x5
            nn.ReLU(inplace=True),  # 使用ReLU激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化层，池化窗口大小为2x2，步幅为2
            nn.Conv2d(6, 16, kernel_size=5),  # 第二个卷积层，输入通道数为6，输出通道数为16，卷积核大小为5x5
            nn.ReLU(inplace=True),  # 使用ReLU激活函数
            nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层，池化窗口大小为2x2，步幅为2
        )
        # 定义分类部分的网络层
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),  # 全连接层，输入大小为16x5x5（展平后），输出大小为120
            nn.ReLU(inplace=True),  # 使用ReLU激活函数
            nn.Linear(120, 84),  # 全连接层，输入大小为120，输出大小为84
            nn.ReLU(inplace=True),  # 使用ReLU激活函数
            nn.Linear(84, num_classes)  # 全连接层，输入大小为84，输出大小为num_classes（分类数）
        )

    def forward(self, x):
        x = self.features(x)  # 特征提取部分的前向传播
        x = torch.flatten(x, 1)  # 将特征展平为一维向量
        x = self.classifier(x)  # 分类部分的前向传播
        return x

# 定义训练函数
def train(model, dataloader, criterion, optimizer):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0  # 初始化累计损失
    correct = 0  # 初始化正确预测数
    total = 0  # 初始化总样本数

    for images, labels in dataloader:
        images = images.to(device)  # 将图像数据移到设备上（CPU或GPU）
        labels = labels.to(device)  # 将标签数据移到设备上

        optimizer.zero_grad()  # 清除前一步的梯度

        outputs = model(images)  # 前向传播，计算输出
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数

        running_loss += loss.item()  # 累加损失
        _, predicted = outputs.max(1)  # 获取预测结果中概率最高的类别
        total += labels.size(0)  # 累加总样本数
        correct += predicted.eq(labels).sum().item()  # 累加正确预测数

    epoch_loss = running_loss / len(dataloader)  # 计算平均损失
    accuracy = correct / total  # 计算准确率

    return epoch_loss, accuracy

# 定义测试函数
def test(model, dataloader, criterion):
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0  # 初始化累计损失
    correct = 0  # 初始化正确预测数
    total = 0  # 初始化总样本数

    with torch.no_grad():  # 不计算梯度
        for images, labels in dataloader:
            images = images.to(device)  # 将图像数据移到设备上
            labels = labels.to(device)  # 将标签数据移到设备上

            outputs = model(images)  # 前向传播，计算输出
            loss = criterion(outputs, labels)  # 计算损失

            running_loss += loss.item()  # 累加损失
            _, predicted = outputs.max(1)  # 获取预测结果中概率最高的类别
            total += labels.size(0)  # 累加总样本数
            correct += predicted.eq(labels).sum().item()  # 累加正确预测数

    epoch_loss = running_loss / len(dataloader)  # 计算平均损失
    accuracy = correct / total  # 计算准确率

    return epoch_loss, accuracy

# 数据路径
train_path = r'D:\人工智能\22336126_李漾_lab_6\data\train'
test_path = r'D:\人工智能\22336126_李漾_lab_6\data\test'

# 数据预处理和转换
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整图像大小到32x32
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 创建数据集
train_dataset = ImageFolder(train_path, transform=transform)
test_dataset = ImageFolder(test_path, transform=transform)

# 创建数据加载器
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 创建CNN模型
num_classes = 5
model = CNN(num_classes)

# 设置设备为CPU
device = torch.device('cpu')
model = model.to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 设置训练参数
num_epochs = 60  # 训练迭代次数
learning_rates = [0.0001, 0.001, 0.01]  # 不同的学习率
train_losses = {lr: [] for lr in learning_rates}  # 用于存储训练损失
train_accuracies = {lr: [] for lr in learning_rates}  # 用于存储训练准确率
test_losses = {lr: [] for lr in learning_rates}  # 用于存储测试损失
test_accuracies = {lr: [] for lr in learning_rates}  # 用于存储测试准确率

# 迭代不同的学习率
for lr in learning_rates:
    model = CNN(num_classes)  # 重置模型
    model = model.to(device)  # 将模型移到设备上
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 使用Adam优化器

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer)  # 训练模型
        test_loss, test_accuracy = test(model, test_loader, criterion)  # 测试模型

        train_losses[lr].append(train_loss)  # 记录训练损失
        train_accuracies[lr].append(train_accuracy)  # 记录训练准确率
        test_losses[lr].append(test_loss)  # 记录测试损失
        test_accuracies[lr].append(test_accuracy)  # 记录测试准确率

        # 打印训练和测试结果
        print(f'Learning Rate: {lr}, Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.1f}')

# 画出损失和准确率曲线图
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
for lr in learning_rates:
    plt.plot(train_losses[lr], label=f'Train (LR={lr})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
for lr in learning_rates:
    plt.plot(train_accuracies[lr], label=f'Train (LR={lr})')
    plt.plot(test_accuracies[lr], label=f'Test (LR={lr})')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
