# 前面讲解了卷积层、池化层，提到了一点padding层，因为用得不多，也讲了非线性激活
# 接下来看一下正则化层，up主说这个是正则化，其实笔者觉得应该说是归一化
# 正则化层其实用的不是很多，其实就是单纯地对输入采用一个正则化，主要是出自有一篇论文，说是采用正则化能够加快神经网络的训练速度，同理，笔者觉得这里说的应该是归一化
# 这里主要讲nn.BatchNorm2d函数，函数的输入主要是num_features，num_features中的C主要来自于输入图像当中的C，也就是channel，后面的其他参数基本上采用默认即可，这个层其实用的不是很多
# 要习惯看官方文档
# 归一化层用的比较少，就不再介绍了，up主在这里称之为正则化
# Recurrent Layers 文字识别中用得比较多，属于一种特定的网络结构，用的概率不是很多
# Transform Layers 也是属于一种特定的结构
# Linear Layers 用到的比较多，并不难，就是要一个in_features，一个out_features，和一个bias偏置，后面再进行讲解
# Dropout Layers 在训练的过程中，会随机地把input tensor中的一些元素变成0，变成0的概率是按p进行设置的，这个层也是在一篇论文中提出来的，主要是防止过拟合
# Sparse Layers 以nn.Embedding为例，其实是用于自然语言处理中
# Distance Function 计算两个值之间的误差，可以设置衡量的方式
# Loss Function 计算误差等

# 本节课重点说线性层，因为比较常用，Recurrent Layers、Transformer Layers、Sparse Layers是在特定的网络中才使用到，有用到的话可以多多关注，Dropout Layers相对来说比较简单，可以自己练手
# Linear Layer其实就是多层感知机中的全连接层
# nn.Linear() 参数in_features可以理解为输入neuron数，out_features可以理解为输出neuron数，bias是否设置为True表示是否需要加上bias
# weight和bias是如何取值的？具体的取值方式和初始化方式在官方文档中有说明，是从某些分布中进行采样和初始化的
# 接下来就来写代码简单看一下
# 我们来看一个典型的网络结构VGG16
# VGG16的末尾存在一个从1 * 1 * 4096到1 * 1 * 1000的变换，其实就是把input_feature设置为4096，output_feature设置为1000
# 这节就来做一个这个东西
# 本节要做的事情是先把一个5 * 5的图像延展成一行25个格子，然后再通过一个线性层变成3个

import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = nn.Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

tudui = Tudui()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # output = torch.reshape(imgs, (1, 1, 1, -1))
    # 最开始用的是上面reshape的方法，但是通过这个例子，引出一个东西叫torch.flatten，flatten是摊平的意思，把他给展平
    # reshape的作用更加强大，可以指定尺寸进行变化，flatten的作用是把它变成一行
    output = torch.flatten(imgs)
    print(output.shape)
    output = tudui(output)
    print(output.shape)

# 至此基本的网络模型讲解完了，其中和特别领域相关的层没有进行特别的讲解
# 网络模型基本算讲解完了，Containers中有一个Sequential中没有讲，但是他使用起来也很简单，下一节来简单说一下
# 我们会自己搭建网络模型了，但是有的时候我们可以直接用pytorch提供的一些网络模型
# 比如说torchvision，有一个torchvision.models，其中提供了很多的网络结构，有的时候可以不用自己写网络，直接调用他的网络结构就可以了
# torchtext好像没有model
# torchaudio也有torchaudio.model，也是可以直接使用的
# 可以看到其实做图像的还是比较多的，提供的模型有分类的，有语义分割的，语义分割做的人还挺多的，也有关于目标检测的、实例分割的、人体关键点检测
# torchvision里面提供的model还是挺多的，有的时候可以直接利用pytorch提供的或者训练好的网络
# 下一节来简单讲解一下这个部分和Containers中的Sequential
