# 我们再来看看如何在之前写的神经网络中用到loss function

import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
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
for data in dataloader:
    imgs, targets = data
    outputs = tudui(imgs)
    # 我们看下outputs和targets长什么样，看选择什么样的损失函数
    # print(outputs)
    # print(targets)
    result_loss = loss(outputs, targets)
    print(result_loss)
    # 我们用代码来看一下如何采用反向传播，一定要注意反向传播的对象是loss求过之后的变量，不能对loss采用backward，用了backward后才会有梯度，有梯度以后才能采用合适的优化器来对参数进行优化，以降低loss
    result_loss.backward()
    print("ok")
    
# 下节课要介绍的内容就是选择合适的优化器，这些优化器会利用梯度来对神经网络中的参数进行更新