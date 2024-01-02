# transform主要是对图片进行一些变化
# Compose就是把许多图像操作结合起来
# ToTensor就是把一个PILImage或者把numpy的array转换成tensor
# ToPILImage转换成PIlImage
# Normalize标准化
# transforms就是把图片输入而后经过一系列工具的处理进而输出一个想要的结果
from PIL import Image
from torchvision import transforms
import cv2
from torch.utils.tensorboard import SummaryWriter

# python的用法 -> tensor数据类型
# 通过transforms.ToTensor去看两个问题
# 2、为什么我们需要Tensor数据类型
# 因为Tensor类型里面有很多用于deeplearning的属性，例如反向传播，梯度，梯度方法等等

img_path = "dataset/train/ants/0013035.jpg"
img = Image.open(img_path)

# 如果想用numpy.ndarray类型的话，可以使用以下读取方式
cv2_img = cv2.imread(img_path)

writer = SummaryWriter("logs")

# 1、transforms该如何使用（python）
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)

writer.add_image("tensor_img", tensor_img)
writer.close()
