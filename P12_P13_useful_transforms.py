# 要想更好地使用他，需要关注三个点，即输入、输出和作用
# 输入和输出是比较容易错的点，因为图片有不同的格式
# PIL Image.open()
# tensor ToTensor()
# narrays cv.imread()

from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
img = Image.open("dataset/train/ants/0013035.jpg")
print(img)

# ToTensor的使用
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize的作用是随着平均值和标准差归一化一个tensor数据类型的image
# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)

trans_norm = transforms.Normalize([1, 3, 5], [3, 2, 1])
img_norm = trans_norm(img_tensor)
writer.add_image("Normalize", img_norm, 1)

trans_norm = transforms.Normalize([1, 2, 3], [3, 2, 1])
img_norm = trans_norm(img_tensor)
writer.add_image("Normalize", img_norm, 2)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL -> totensor -> img_resize tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)
print(type(img_resize))

# Compose - resize - 2
# Compose()中的参数需要是一个列表，且参数需要是transforms类型
# 所以，Compose([transforms参数1, transforms参数2,...])
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL -> tensor
# Compose中要注意前后两个操作的输入输出数据类型是否匹配
# 后一个参数的输入与前一个参数的输出要匹配
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop((500, 700))
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCropHW", img_crop, i)


writer.close()
# Compose就是把不同的transforms结合在一起
# ToTensor就是把PIL Image类型和numpy.ndarray类型转化为tensor类型
# tensorboard必须是一个tensor数据类型

# 总结：
# 1、要使用的话要关注输入和输出类型
# 2、多看官方文档
# 3、关注方法需要什么参数
# 不知道返回值的时候"print", "print(type())", debug
