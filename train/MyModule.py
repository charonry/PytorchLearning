import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential


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


if __name__ == '__main__':
    myModule = MyModule()
    input = torch.ones(64, 3, 32, 32)
    output = myModule(input)
    print(output)

