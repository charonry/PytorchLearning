from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
"""
tensorboard --logdir=logs --port=6006 (默认端口) --reload_interval=5 (刷新频率)
"""
writer = SummaryWriter("logs")
# 1.标量数值读取
"""
for i in range(100):
    writer.add_scalar("y=3*x", 3*i, i)
"""
# 2.图片信息读取
image_path = r"resource/dataset_group/train/bees/16838648_415acd9e3f.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

writer.add_image("image_show", img_array, 2, dataformats="HWC")


writer.close()