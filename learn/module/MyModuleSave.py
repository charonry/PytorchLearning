import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1:保存网络模型结构和参数
torch.save(vgg16, "./modulesave/vgg16_method1.pth")
# 保存方式2:保存网络模型的参数（以字典形式）(官方推荐)
torch.save(vgg16.state_dict(), "./modulesave/vgg16_method2.pth")


# 方式1存在陷阱
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
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, input):
        x = self.model1(input)
        return x


myModule = MyModule()
torch.save(myModule, "./modulesave/my_module.pth")


