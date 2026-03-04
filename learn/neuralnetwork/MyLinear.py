import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Linear


class MyMoudle(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = Linear(196608, 10)

    def forward(self, input):
        input_flatten = torch.flatten(input)
        print(input_flatten.shape)
        return self.linear(input_flatten)


myMoudle = MyMoudle()

dataset = torchvision.datasets.CIFAR10("./dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

for data in dataloader:
    imgs, target = data
    print(imgs.shape)
    output = myMoudle(imgs)
    print(output.shape)


