# 今天讲解非线性激活
# 打开pytorch官网的torch.nn，能够看到卷积层、池化层我们都讲过，但是下面的padding层是什么呢？
# padding层其实几乎用不到，就是单纯地在input tensor边缘进行各式各样的填充
# 接下来讲非线性激活，我们可以看到有两个不同的非线性激活，非线性激活主要是为了给神经网络中引入一些非线性特征
# 其中nn.ReLU()是比较常见的，需要注意输入的形状，输入形状为(N, *)的意思是元组中第一个参数需要为N，也就是batch_size，后面的形状不进行限制，此处需要说明的是最新版本的pytorch输入形状仅需(*)即可，不需要再指定batch_size了
# nn.ReLU()中有一个inplace参数，是是否替换的意思，若为True，则有返回值，并在输入tensor进行替换，若为False，则依然有返回值，并不替换输入tensor，一般把inplace设置为False
# nn.Sigmoid()也比较常用
# 可以看到非线性激活的使用不是很难，此处就用nn.ReLU()举例子

import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output

tudui = Tudui()
output = tudui(input)
print(output)

writer = SummaryWriter("../logs_relu")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output = tudui(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()

# 非线性变换的主要目的是给网络引入非线性特征，因为非线性越多，才能训练出符合各种曲线或者说符合各种特征的模型，如果大家都是线性的话，那模型的泛化能力就不够好
# 非线性其实不难，只要会了一个，另外一个无非是非线性的处理公式不一样
