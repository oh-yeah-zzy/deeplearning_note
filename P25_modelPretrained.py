# 这节课来讲解一下pytorch提供的一些网络模型
# 包括这些网络模型应该如何去使用，以及如果想对这些网络模型进行一些修改，我们该如何操作
# torchvision中有一些关于图像的模型
# torchvision.models中的模型有不同的分类，例如说有关于分类的model，还有语义分割的model，还有目标检测的model
# 这节主要讲解简单的分类模型，数据集还用CIFAR10模型，这个数据集本身也是用于分类的
# 我们用最常用的VGG
# 我们来讲解一下VGG16，其中最常用的一般是VGG16或VGG19
# torchvision.models.vgg16中主要有两个参数
# pretrained 如果为True的话会使用在某个数据集（对于VGG16来说是ImageNet）上预训练好的模型，若为False则模型中的参数仅是一些初始化的参数，没有进行任何的训练
# progress 若为True则会显示一个下载进度条，若为False则不显示
# 我们想要测试pretrained参数的作用，那我们就下载ImageNet数据集，然后看pretrained分别为True和False时model在这个数据集上的效果
# 找到torchvision.datasets，然后找到这个数据集
# 这个数据集介绍的比较简单，用起来也不难
# root是下载的路径，split是指要训练集还是验证集
# transform是是否要在图像上进行变换
# target_transform是是否要在target上进行变换
# load是function如何去加载这个数据集

import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

# 首先来下载数据集
# 新版本的torchvision不需要download这个参数，没下载会直接下载，已经下载就不用重新下载
# train_data = torchvision.datasets.ImageNet("./data", split="train", download=True,
#                                            transform=torchvision.transforms.ToTensor())
# 由于ImageNet数据集太大，无法通过下载数据集来验证pretrained这个参数
# 那就换种方式，通过观察pretrained参数分别为True和False时下载的网络模型的VGG16的参数有什么差别来判断

# 这里的pretrained和up讲得不一样了，因为新版的设置不是pretrained，而是weights了
# 当weights为None的时候，仅是加载这个网络，是不需要下载的
# 但是当weights设置为预训练的模型的时候，是需要去下载网络参数的
vgg16_false = torchvision.models.vgg16(weights=None)
vgg16_true = torchvision.models.vgg16(weights='DEFAULT')
print('ok')

print(vgg16_true)
# 打印出来会有很多层，我们主要关注最后面一个，最后面一个线性层的输出是1000，也就是说VGG16这也是一个分类模型，他能够分出的类别有1000个
# 他其实是在ImageNet中进行一个训练的，找到数据集进行查看，能够看到数据集也是有1000个类别的

# CIFAR10仅把数据分为了10类，但是VGG16把数据分为了1000类，那我们如何去应用这个网络模型呢？
train_data = torchvision.datasets.CIFAR10("./data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
# 一种简单的方式是把最后一层的out_features的1000改成10，这种方式确实很简单
# 或者另外一种简单的方式是在最后一层的线性层下面再加一层，让他的输入是1000，让他的out_features是10，同样可以完成这个任务
# 所以现在分两步走，一个是说如何利用现有的网络去改动它的结构，这样就可以避免写这个VGG16
# 我们会发现很多的框架都会把VGG16当作前置的一个网络结构，一般说就是用VGG16来提取一些特殊的特征
# 然后在VGG16后面再加一些网络结构去实现一些功能，所以说VGG16还是挺重要的，我们学习如何在现有的一些网络结构进行修改是很有必要的

# 我们以前面的vgg16_true进行讲解，我们现在想给他添加一层，让他的in_features是1000，out_features是10，那我们就可以完成这个网络模型了
# 那我们就在这个vgg16_true进行修改
# add_module的第二个参数module，一个层是一个module，一个被Sequential包起来的若干个层也是一个module
vgg16_true.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

# 假设说不想把层单独作为一个module，而是想把层加到vgg16的classifier中，那该怎么办呢？
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

# 这样我们就学会了如何在现有的网络模型中去加一些东西
# 假设此时又有人说不想加，想修改，修改怎么改呢？
# 因为vgg16_true已经修改过了，就能vgg16_false来距离
print(vgg16_false)
# 例如说此时想把最后一层修改为in_features是4096，out_features是10
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)

# 现在我们学会了如何加载一些现有的pytorch的网络模型，也学会了如何对网络模型中的结构进行修改，包括添加一些自己想要的网络模型结构
# 删除网络中的某些层可以直接使用nn.Sequential()将该层直接设置为空即可
# 冻结网络中的某些层就直接使该层的requires_grad=False即可，例如{net}.weight.requires_grad=False，这里的net是一种泛指