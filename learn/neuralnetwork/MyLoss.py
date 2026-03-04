import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

input = torch.tensor([1, 2, 3], dtype=torch.float32)
target = torch.tensor([1, 2, 5], dtype=torch.float32)

# 默认是mean
loss = L1Loss(reduction='mean')
result = loss(input, target)
print(result)

loss_mes = MSELoss()
result_mes = loss_mes(input, target)
print(result_mes)

# 交叉熵
x = torch.tensor([0.1, 0.3, 0.2])
print(x.shape)
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
print(x.shape)
loss_cross = CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)
