# 本节课主要讲解神经网络中一些基本神经结构的使用
# 本节课主要讲解Convolution Layers
# 1d代表是一维的，2d代表是二维的，图片就是典型的二维的，主要讲解nn.Conv2d
# torch.nn和torch.nn.functional的区别是torch.nn是torch.nn.functional的一个封装
# 如果想要细致地了解一些卷积的操作，还是要详细讲解一下torch.nn.functional
# 其实只要学习torch.nn就可以了，torch.nn.functional其实是不需要了解的
# torch.nn.functional里面其实有许多和torch.nn对应的模块
# Conv2d中的一些参数解释如下：
# input 输入
# weight 权重，其实更专业的说法可以叫做卷积核，参数设置中的groups一般取为1
# bias 偏置
# stride 步进，即在图像中卷积核本次和下一次移动到的位置之间的长度，注意，stride可以设置为一个tuple，来分别控制横向和纵向的步长，如果是一个number则横向和纵向的stride是一样的

#卷积计算的时候其实就是对应位置的数字相乘然后全部加在一起

import torch
import torch.nn.functional as F

input = torch.tensor([[1, 2, 0, 3, 1], 
                      [0, 1, 2, 3, 1], 
                      [1, 2, 1, 0, 0], 
                      [5, 2, 3, 1, 1], 
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

# pytorch提供的一个尺寸变换的工具
input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

print(input.shape)
print(kernel.shape)

# kernel就是weight，padding和dilation后面再进行讲解
output = F.conv2d(input, kernel, stride=1)
print(output)

output2 = F.conv2d(input, kernel, stride=2)
print(output2)

# 接下来讲解padding
# padding主要是在图像的左右两边进行填充，它也可以是一个数或者一个元组，默认是不进行填充的
# padding出来的值一般是0
output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(output3)
