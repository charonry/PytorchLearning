import torchvision
from torch.utils.data import DataLoader
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
# print(vgg16_false)
# print("*"*30)
# print(vgg16_true)

dataset = torchvision.datasets.CIFAR10("./dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)
# 新增
# vgg16_true.add_module('add_linear', nn.Linear(1000, 10))  # 加载classifier之后
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))  # 加载classifier内
# print(vgg16_true)

# 修改
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)


