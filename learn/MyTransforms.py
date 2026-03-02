from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import cv2

writer = SummaryWriter("logs")

image_path = r"resource/dataset_group/train/bees/95238259_98470c5b10.jpg"
img = Image.open(image_path)
transor_trans = transforms.ToTensor()
transor_img = transor_trans(img)
writer.add_image("tensor_img", transor_img)
writer.close()

print(transor_img)


