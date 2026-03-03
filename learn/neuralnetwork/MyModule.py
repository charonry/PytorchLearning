import torch


class MyFirstModuleDemo(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


demo_module = MyFirstModuleDemo()
x = torch.tensor(1)
output = demo_module(x)
print(output)
