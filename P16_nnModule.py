# 关于神经网络的工具主要在torch.nn里面，来自nerual network的缩写
# Containers: 中文翻译是容器，但是更好的解释是骨架，给神经网络定义了一些结构，主要往这些结构中添加一些不同的内容就能够组成一个神经网络，这节主要学习这个
# 像后面的一些卷积层、池化层、padding、非线性激活、正则化层是组成神经网络中一些核心的操作部分
# Containers中有六个模块，Module是我们最常用的模块，所有神经网络模块的基类
# Module定义了一个基类，继承它就是拿他的模板过来用，对于不满意的部分进行修改
# Module中有两个方法，一个是__init__，另一个是forward
# __init__函数是做初始化，首先要写super(Model, self).__init__()，这个语句的意思是调用Model父类的__init__函数，然后再写自己想写的
# forward函数是向前的，就是我们给一个input，然后经过神经网络中的一个forward，然后给一个output，这个叫做前向传播，还有一个对应的反向传播，也就是backward
# forward函数定义了一个self和一个x，相当于输入是x，其实也可以改成任意的，例如说input
# forward定义了每次的计算图，在每个子类中都应该被重写，接下来就来写一个例子看一下

import torch
from torch import nn

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

tudui = Tudui()
x = torch.tensor(1.0)
output = tudui(x)
print(output)

# 本节课主要讲解了nn.Module的使用，后面会讲解一些常用的层，例如卷积层
