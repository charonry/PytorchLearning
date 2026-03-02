from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

writer = SummaryWriter("logs")

image_path = r"resource/dataset_group/train/bees/16838648_415acd9e3f.jpg"
img = Image.open(image_path)
# ToTensor
trans_toTensor = transforms.ToTensor()
img_tensor = trans_toTensor(img)
writer.add_image("ToTensor", img_tensor)
print(img_tensor[0][0][0])
# Normalize（归一化）
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)
# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
print(img_resize.size)
img_resize = trans_toTensor(img_resize)
writer.add_image("Resize", img_resize, 1)
# Compose
trans_resize_2 = transforms.Resize(200)
trans_compose = transforms.Compose([trans_resize_2, trans_toTensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 2)
# RandomCrop
trans_random = transforms.RandomCrop((300, 200))
trans_compose_2 = transforms.Compose([trans_random, trans_toTensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)
writer.close()
