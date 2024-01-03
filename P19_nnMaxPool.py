# 这节课来讲解池化层
# 最大池化有的时候也被叫做下采样，maxUnpool一般叫做上采样，还有一些平均池化、自适应最大池化等等
# 其实最常用的还是MaxPool2d
# 参数其实很简单，只要一个kernel_size，就是池化核
# kernel_size 设置的是一个用来取最大值的窗口，设置方式和卷积层一样
# stride 设置方式和卷积层一样，是横向和纵向的一个步长，但是默认值和卷积层不一样，卷积层的默认值是1，池化层的默认值是kernel_size的大小
# padding 设置方式和卷积层是一样的
# dilation 卷积步长，即卷积核中一个元素和其相邻元素之间的距离，也被称为空洞卷积，因为当被设置为大于0的数的时候就相当于是卷积核中间有了个洞，一般不进行设置
# return_indices 一般情况下用得非常少，这里不做了解
# ceil_mode 当设置为True时，会设置为ceil模式来计算输出的shape，而不是floor模式，当池化核覆盖的范围超出输入的大小的时候，是放弃这个池化还是说在能覆盖的范围内进行池化，就是由ceil_mode来决定的，当ceil_mode为True时，需要保留这个覆盖的范围，即在能覆盖的范围内进行池化，若ceil_mode为False时，就不进行保留，即放弃这个池化，需要注意ceil_mode默认为False
# 什么是ceil模式和floor模式呢？floor是向下取整，ceil是向上取整
# 池化是个什么意思？例如最大池化会取池化核覆盖的范围当中最大的数值
# 最大池化一般只需要设置kernel_size

import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, download=True, transform= torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

# 此处参数加上dtype的原因是否则会被认为是long型，进而报出bug
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)

input = torch.reshape(input, (-1, 1, 5, 5))
print(input.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool(input)
        return output

tudui = Tudui()
output = tudui(input)
print(output)

writer = SummaryWriter("../logs_maxpool")
step = 0

# 池化中没有多个channel，如果是三个channel的图像池化后还是三个channel，所以不用像卷积中那样如果想要在tensorboard中看的话还需要reshape
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = tudui(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()

# 为什么要进行最大池化？最大池化的作用是什么？
# 最大池化的目的是想保留输入的特征，但同时减少数据量
# 减少数据量对于整个网络来说进行计算的参数就变少了，就会训练的更快
# 最大池化后图像会模糊一些，但是会尽量保留输入图像中的一些信息
# 主要还是因为通过这样的池化训练的数据量会大大的减小，所以很多网络中都会一层卷积后一层池化，池化后再来一层非线性激活
# 非线性激活就是下节课要讲的内容
