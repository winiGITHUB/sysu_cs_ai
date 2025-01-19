import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels or stride != 1 else None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class My_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(My_ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# 定义超参数
num_epochs = 1

# 实验记录
results = {}

# 定义不同的实验参数
learning_rates = [0.01, 0.1, 0.001]
batch_sizes = [32, 64, 128]
model_variants = {
    "ResNet-10": [1, 1, 1],
    "ResNet-18": [2, 2, 2],
    "ResNet-34": [3, 4, 6]
}

# 数据预处理和加载
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model_name, num_blocks in model_variants.items():
    for lr in learning_rates:
        for batch_size in batch_sizes:
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
            
            # 实例化模型、损失函数和优化器
            model = My_ResNet(BasicBlock, num_blocks).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=lr)

            # 训练和测试函数
            def train(model, device, train_loader, optimizer, criterion):
                model.train()
                train_loss = 0
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                return train_loss / len(train_loader.dataset)

            def test(model, device, test_loader, criterion):
                model.eval()
                test_loss = 0
                correct = 0
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        test_loss += criterion(output, target).item()
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                test_loss /= len(test_loader.dataset)
                accuracy = 100. * correct / len(test_loader.dataset)
                return test_loss, accuracy

            # 记录实验结果
            experiment_name = f"{model_name}_LR_{lr}_BS_{batch_size}"
            train_losses, test_losses, test_accuracies = [], [], []
            for epoch in range(num_epochs):
                train_loss = train(model, device, train_loader, optimizer, criterion)
                test_loss, accuracy = test(model, device, test_loader, criterion)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                test_accuracies.append(accuracy)
                print(f"Experiment: {experiment_name}, Epoch: {epoch}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, Accuracy: {accuracy:.2f}%")
            
            results[experiment_name] = {
                "train_losses": train_losses,
                "test_losses": test_losses,
                "test_accuracies": test_accuracies
            }

# 绘制结果
for key, value in results.items():
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(value["train_losses"], label='Training Loss')
    plt.plot(value["test_losses"], label='Testing Loss')
    plt.title(f'{key} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(value["test_accuracies"], label='Testing Accuracy')
    plt.title(f'{key} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
