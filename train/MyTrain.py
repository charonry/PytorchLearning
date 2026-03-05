import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from MyModule import *
from torch import nn

train_data = torchvision.datasets.CIFAR10("./dataset", train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)

test_data = torchvision.datasets.CIFAR10("./dataset", train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True)
test_data_size = len(test_data)

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建神经网络
myModule = MyModule()
# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 优化器
learn_rate = 1e-2
optimizer = torch.optim.SGD(myModule.parameters(), lr=learn_rate)
# 设置训练网络的一些参数
total_train_step = 0
total_test_step = 0
epoch = 10
# 添加tensorboard
writer = SummaryWriter("logs")
for i in range(epoch):
    print(f"----------第{i+1}轮开始训练----------")
    # 训练步骤
    for data in train_dataloader:
        imgs, targets = data
        outputs = myModule(imgs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"本轮批次训练次数：{total_train_step}；Loss：{loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
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



