import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch import nn
import time

# 定义训练设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# 创建模型
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 2, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(64*4*4, 64),
            Linear(64, 10)
        )

    def forward(self, input):
        x = self.model1(input)
        return x


train_data = torchvision.datasets.CIFAR10("./dataset", train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)

test_data = torchvision.datasets.CIFAR10("./dataset", train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True)
test_data_size = len(test_data)

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建神经网络
myModule = MyModule()
myModule.to(device)
# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)
# 优化器
learn_rate = 1e-2
optimizer = torch.optim.SGD(myModule.parameters(), lr=learn_rate)
# 设置训练网络的一些参数
total_train_step = 0
total_test_step = 0
epoch = 10
# 添加tensorboard
writer = SummaryWriter("logs")
start_time = time.time()
for i in range(epoch):
    print(f"----------第{i+1}轮开始训练----------")
    # 训练步骤
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = myModule(imgs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(f"时间差{end_time-start_time}")
            print(f"本轮批次训练次数：{total_train_step}；Loss：{loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = myModule(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            # 计算正确率
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print(f"整体测试集上的Loss：{total_test_loss}")
    accuracy_rate = (total_accuracy/test_data_size).item()
    print(f"整体测试正确率：{accuracy_rate}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", accuracy_rate, total_test_step)
    total_test_step += 1

    # 保存模型
    torch.save(myModule, f"./modulesave/my_module_{i}.pth")

writer.close()



