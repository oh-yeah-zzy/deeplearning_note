# 本节主要讲解torch.nn中Conv2d的使用，其他的卷积层使用方法差不多
# in_channels 输入通道数，彩色图像一般是三个，即输入图片的channel数
# out_channels 输出通道数，后面会有实际案例讲解，输出图片的channel数，例如in_channels=1, out_channels=2时，会生成两个卷积核，对同一张图像进行卷积，这样子会卷积出两个结果，然后叠起来就是两个out_channel了，很多算法其实都会不断地增加channel数
# kernel_size 卷积核的大小，可以是一个数或者一个元组，譬如说3的话就是3*3的卷积核，在搭建卷积层的时候，只需要定义kernel_size的大小尺寸就行了，其中的数不需要进行设置，实际在训练的过程中会对kernel中的值进行不断地调整
# stride 卷积过程中步进的大小
# padding 卷积过程中需要对原始图像进行padding的大小
# dilation 其实就是卷积核一个对应位的距离，比较难理解，后面会找个图片，卷积步长，可以用来做空洞卷积，但是几乎不常用
# groups 一般设置为1，很少需要改动，分组卷积的时候需要进行修改，但是几乎遇不到
# bias 偏置，一般设置为True，其实就是对卷积后的结果是否加减一个常数
# padding_mode 就是当选择padding的时候以什么样的模式进行填充
# 只有前三个参数是必须要设置的，其他参数都有默认值
# 在日常使用中，其实常用的只有前面5个参数，需要自己进行设置

import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(), 
                                      download=True)
dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

tudui = Tudui()

# 想看神经网络结构的话这样直接print就可以了
print(tudui)

writer = SummaryWriter("../logs")
step = 0
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    print(imgs.shape)
    print(output.shape)
    # 输入的尺寸 torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)
    # 卷积后输出的尺寸 torch.Size([64, 6, 30, 30])，但是由于6个通道的图像tensorboard无法显示，我们想把他变成[xxx, 3, 30, 30]
    # 当reshape的形状的第一个值不知道是多少的时候，可以直接写-1，它会根据后面的形状自行计算
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)

    step = step + 1 

# 学完这些之后，我们找一些卷积层来看一下，例如VGG16
# VGG16中有个很有意思的一点，可以用来理解卷积核是如何工作的，VGG16的输入图像是224 * 224 * 3的，经过第一个卷积层以后图像是224 * 224 * 64的，需要一个什么样的卷积核才能把3变成64呢，此处应该使用x * y * 3 * 64的卷积核(这里的x和y表示的是未知数的意思，因为笔者在写这段的时候也不知道VGG16的卷积核是什么形状的)，卷积核三个一组，分别对三个通道卷积后叠加起来，进一步需要64组卷积核来输出64个channel，其实可以简单理解为输出通道数等于卷积核个数
# 注意官网上torch.nn.Conv2d中Hout和Wout的计算公式，看论文的时候如果别人没有给stride和padding的公式的话，就需要自己根据这个公式推导

