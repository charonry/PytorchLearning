import torch
import torchvision
# 解决方式1陷阱
from MyModuleSave import MyModule

# 方式1 =》保存方式1 加载模型
save_module_1 = torch.load("./modulesave/vgg16_method1.pth")
# print(save_module_1)
# 方式2 =》保存方式2 加载模型
# 2.1：只有参数
save_module_2 = torch.load("./modulesave/vgg16_method2.pth")
# print(save_module_2)
# 2.2：结构+参数
vgg16 = torchvision.models.vgg16(pretrained=False)
# print(vgg16)
vgg16.load_state_dict(torch.load("./modulesave/vgg16_method2.pth"))
# print(vgg16)


# 方式1陷阱
my_module_1 = torch.load("./modulesave/my_module.pth")
print(my_module_1)
