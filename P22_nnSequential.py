# 终于快把神经网络的基础结构讲完了
# 这节课主要讲Containers中的Sequential
# 但是Sequential其实很简单，把网络结构放到Sequential里面，然后一个model()，给一个input，就会按照顺序执行下去
# Sequential的好处是代码写起来会比较简洁易懂
# 因为Sequential很简单，所以这节主要讲如果搭建网络模型使用了Sequential代码是否会变得更简单容易、容易管理
# 所以这节写一个比较简单的网络模型
# 因为之前用的数据集是CIFAR10()，所以写一个用来对CIFAR10()进行分类的简单神经网络
# up主从网上搜索了一个CIFAR-10 model structure来进行实现
# 最大池化不改变channel数
# 一般来说padding设置成kernel_size // 2就是保持图像原来的大小

import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(3, 32, 5, padding=2)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = MaxPool2d(2)
        # 之前有讲过一个torch.flatten，但是其实flatten也有像卷积层、池化层这样的flatten层，位于torch.Flatten中
        self.flatten = Flatten()
        # 有一种说法是，一般线性层都会用1*1的卷积，因为卷积操作更快
        self.linear1 = Linear(1024, 64)
        self.linear2 = Linear(64, 10)

    # 激活函数不在__init__里面写，在forward里面写也是可以的，但是不是所有的网络都有激活函数
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

tudui = Tudui()
print(tudui)

# 在实践过程中我们该如何检查网络的正确性呢？
# 其实要关注的就是我们的数据经过这个网络后能否正确地得到我们想要的输出
# 因为写完网络后，即使网络错误也能被创建，所以一般写完网络后要检查网络的正确性
# 一般通过创建一个假想的输入来进行检查
# torch中有提供一些简单的创建tensor的方法，比如说都是0、都是1、都是随机数

input = torch.ones((64, 3, 32, 32))
output = tudui(input)
print(output.shape)
# 一张图片产生的输出是10，但是有batch_size 64，所以产生的output.shape是(64, 10)
# 如果后面两个linear层的尺寸不会算，就把后面两个linear层去掉，输出看去掉后的网络经过输入输出的数据的格式是什么样的，主要需要关注的是batch_size后面的尺寸
# 接下来要讲的就是之前一直提的Sequential

class TuduiSequential(nn.Module):
    def __init__(self):
        super().__init__()
        # Sequential还会自动给层按从前往后的顺序编号，编号从0开始
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

tudui = TuduiSequential()
print(tudui)
input = torch.ones((64, 3, 32, 32))
output = tudui(input)
print(output.shape)

# 模型除了print这种可视化方式以外，其实tensorboard也可以对其进行可视化

writer = SummaryWriter("../logs_seq")
# 这里的graph是计算图的意思
writer.add_graph(tudui, input)
writer.close()

# 至此整个网络的搭建讲解完了
# 接下来会讲解如何使用一些常见的、pytorch给我们提供的网络
