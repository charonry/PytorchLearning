import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class MyModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, input):
        # output = self.relu(input)
        output = self.sigmoid(input)
        return output


myModule = MyModule()
"""
input = torch.tensor([[1, -0.5],
                      [-1, 3]], dtype=torch.float32)

input = torch.reshape(input, (-1, 1, 2, 2))
print(input)


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
    writer.add_images("input_sigmoid", imgs, global_step=step)
    output = myModule(imgs)
    writer.add_images("output_sigmoid", output, global_step=step)
    step += 1

writer.close()

