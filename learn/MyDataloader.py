import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())
# batch_size：每批次获取样本个数
# drop_last:如果最后一组不足batch_size是否舍弃
test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=True, num_workers=0, drop_last=False)
img, target = test_data[0]
print(img.shape, target)
print("*"*50, len(test_data), len(test_loader))

writer = SummaryWriter("logs")


for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images(f"epoch:{epoch}", imgs, step)
        step += 1
        print(imgs.shape, targets)
writer.close()
