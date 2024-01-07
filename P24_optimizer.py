# 这一节主要讲解优化器
# 前面讲到我们可以调用损失函数的backward，反向传播可以求出需要调整参数的梯度
# 有了梯度，我们可以利用优化器根据梯度对参数进行调整，以降低整体误差
# 这一节主要讲解优化器
# 所有的优化器都集中在torch.optim

# 优化器的第一个参数是模型的参数
# lr是learning rate，就是学习速率
# 后面的一些其他参数都是特定的算法中要设置的
# 不同的优化器后面的参数除了前两项后面的参数都是有所差距的
# 如果只是入门的话就设置param和lr就可以了，后面的参数就采用默认的参数或者别人使用的一些参数就可以了
# 接下来写代码来看一下

import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./data", train=False, transform=torchvision.transforms.ToTensor(), 
                                       download=True)
dataloader = DataLoader(dataset, batch_size=1)

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

loss = nn.CrossEntropyLoss()
tudui = TuduiSequential()
# lr不能设置太大，也不能设置太小，太大模型训练起来很不稳定，太小模型训练起来比较慢，一般情况下推荐一开始采用比较大的lr，后面采用比较小的lr
optim = torch.optim.SGD(tudui.parameters(), lr = 0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        result_loss = loss(outputs, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss = running_loss + result_loss
    print(running_loss)

# 能够看到随着epoch的增加，running_loss在减小
# running loss是每一轮训练中对于每张图片的loss的总和
# 优化器的套路就是先设置一个优化器，然后把优化器的梯度清零，然后使用backward求梯度，然后采用step对模型参数进行调优
# 相当于是把模型的训练也进行了说明，在实际的训练过程中，epoch应该是成百上千的，这里用的20其实算少的