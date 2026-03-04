import torch
from torch import nn
from torch.nn import MaxPool2d
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxPool = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxPool(input)
        return output


myModule = MyModule()

"""
input = torch.tensor([[1, 2, 0, 3, 1],
                     [0, 1, 2, 3, 1],
                     [1, 2, 1, 0, 0],
                     [5, 2, 3, 1, 1],
                     [2, 1, 0, 1, 1]], dtype=torch.float32)

input = torch.reshape(input, (-1, 1, 5, 5))

output = myModule(input)
print(output)
"""

dataset = torchvision.datasets.CIFAR10("./dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)

writer = SummaryWriter("logs")

step = 0
for data in dataloader:
    imgs, target = data
    writer.add_images("input_maxpool", imgs, step)
    output = myModule(imgs)
    writer.add_images("output_maxpool", output, step)
    step += 1

writer.close()