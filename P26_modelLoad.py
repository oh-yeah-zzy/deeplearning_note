# 这里写的是模型加载的代码

import torch
import torchvision
from torch import nn

# 用保存方式一保存的模型可以用这种方式加载，加载方式1
model = torch.load("vgg16_method1.pth")
print(model)
# 这里print出来的是模型的结构，但是其实模型的参数也被保存下来了，可以debug看一下

# 用保存方式二保存的模型可以用这种方式加载，加载方式2
model = torch.load("vgg16_method2.pth")
print(model)
# 方式二打印出来的是一个一个的参数，并没有像方式一一样的有网络结构
# 保存的是一种字典形式了，不再是一个网络模型了
# 如果要恢复成网络模型应该怎么办？那就要新建网络模型结构
vgg16 = torchvision.models.vgg16(weights=None)
vgg16.load_state_dict(model) # 等效于vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)

# 关于保存方式一的陷阱
class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        
    def forward(self, x):
        x = self.conv1(x)
        return x

model = torch.load("tudui_method1.pth")
print(model)
# 如果不写class Tudui那个类，则会报错AttributeError: Can't get attribute 'Tudui' on <module '__main__' from 'G:\\tudui\\P26_modelLoad.py'>
# 需要去把之前的网络结构去复制过来，但是不需要去创建了
# 也就是说需要有模型的定义，应该是为了确保加载的网络模型是想要的网络模型
# 用现有的网络模型是看不出这个陷阱的，但是用自己的网络模型就可以看出来这个陷阱
# 真实写项目的时候不一定会遇到这个陷阱，因为一般会加一句 from P26_modelSave import *，这个时候就可以不写class Tudui那个类
# 简而言之就是用方式一的话要让程序能够访问到模型定义的方式