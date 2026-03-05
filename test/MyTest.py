from PIL import Image
import torchvision
import torch
from train import MyModule
"""
使用 torch.load(PATH) 加载一个完整模型时，PyTorch 并不只是保存了参数，它还保存了该类定义时的路径信息。
我MyModule在train文件夹，MyTest在test文件夹下 需要将强制映射
所以推荐的“仅保存权重”方式
"""
import sys
sys.modules['MyModule'] = MyModule


image_path = r"resource/dataset_group/train/bees/16838648_415acd9e3f.jpg"
image = Image.open(image_path)


transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                           torchvision.transforms.ToTensor()])

image = transform(image)


has_module = torch.load("./modulesave/my_module_7.pth")
print(has_module)

image = torch.reshape(image, (1, 3, 32, 32))
has_module.eval()
with torch.no_grad():
    output = has_module(image)
print(output)
# 百分分出错，因为图片是蜜蜂，CIFAR10没有这个分类。这只是样例
print(output.argmax(1))
