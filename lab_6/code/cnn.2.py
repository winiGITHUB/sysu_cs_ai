import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.BatchNorm2d(6),  # 添加批归一化层
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),  # 添加批归一化层
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # 添加Dropout层
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # 添加Dropout层
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    accuracy = correct / total

    return epoch_loss, accuracy

def test(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    accuracy = correct / total

    return epoch_loss, accuracy

# 数据路径
train_path = r'D:\人工智能\22336126_李漾_lab_6\data\train'
test_path = r'D:\人工智能\22336126_李漾_lab_6\data\test'

# 数据预处理和转换
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),      # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建数据集
train_dataset = ImageFolder(train_path, transform=transform)
test_dataset = ImageFolder(test_path, transform=transform)

# 创建数据加载器
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 设置设备
device = torch.device('cpu')

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 设置训练参数
num_epochs = 60
learning_rates = [0.0001, 0.001, 0.01]
train_losses = {lr: [] for lr in learning_rates}
train_accuracies = {lr: [] for lr in learning_rates}
test_losses = {lr: [] for lr in learning_rates}
test_accuracies = {lr: [] for lr in learning_rates}

# 在每个学习率下训练模型
for lr in learning_rates:
    model = CNN(num_classes=5)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # 每20个epoch学习率降低为原来的0.1倍

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer)
        test_loss, test_accuracy = test(model, test_loader, criterion)

        scheduler.step()  # 更新学习率

        train_losses[lr].append(train_loss)
        train_accuracies[lr].append(train_accuracy)
        test_losses[lr].append(test_loss)
        test_accuracies[lr].append(test_accuracy)

        print(f'Learning Rate: {lr}, Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# 绘制损失和准确率曲线图
plt.figure(figsize=(12, 5))
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
