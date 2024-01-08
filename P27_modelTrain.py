# 这节课讲解完整的模型训练套路
# 以CIFAR10作为这节课的例子，来完成对这个数据集的分类问题
# 这个数据集总共有10个类别，对于网络来说，这是一个10分类的问题
# 如果觉得数据集太简单，会在最后的时候来看一下github上优秀的代码或项目
# 模型训练套路讲完会讲GPU训练，然后再讲完整的模型验证套路

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

# 网络模型在P27_model.py中，所以需要import该文件，*代表引入所有东西
from P27_model import *

# 第一步是准备数据集
train_data = torchvision.datasets.CIFAR10(root="./data", train=True, transform=torchvision.transforms.ToTensor(),
                                         download=True)
test_data = torchvision.datasets.CIFAR10(root="./data", train=False, transform=torchvision.transforms.ToTensor(),
                                          download=True)

# 如果想看一下训练数据集和测试数据集有多少张
train_data_size = len(train_data)
test_data_size = len(test_data)
# 这种写法是python中一种常用的写法，被称为字符串格式化
# python3.6新增了一种写法，也可以写为print(f"训练数据集的长度为: {train_data_size}")
print("训练数据集的长度为: {}".format(train_data_size))
print("测试数据集的长度为: {}".format(test_data_size))

# 现在获得了数据集，也知道了数据集的长度
# 现在利用Dataloader来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 现在准备好了数据集，也用了DataLoader对数据集进行了加载
# 下面一步可以搭建神经网络
# 因为CIFAR10中有10个类别，所以这个网络应该是一个10分类的网络
# 这个网络我们前面已经讲解过了

# 很多人喜欢把神经网络放到一个单独的python文件中，叫做model，然后直接import网络，这里为了规范我们也这么做

# 创建网络模型
tudui = Tudui()

# 接下来就可以创建损失函数
loss_fn = nn.CrossEntropyLoss()

# 定义优化器
# learning rate这样提出来方便修改
# learning_rate = 0.01
# 1e-2的意思是1 x (10)^(-2) = 1/100 = 0.01，这样的写法方便改learning rate的大小
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# total_train_step 记录训练的次数
total_train_step = 0
# total_test_step 记录测试的次数
total_test_step = 0
# epoch 训练的轮数
epoch = 10

# 开始训练
for i in range(epoch):
    print("------------第 {} 轮训练开始------------".format(i+1))
    
    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)
        
        # 优化器优化模型
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播得到每个参数节点的梯度
        loss.backward()
        # 调用优化器对参数进行优化
        optimizer.step()
        
        # 进行了一次优化，也就相当于进行了一次训练
        total_train_step = total_train_step + 1
        # print("训练次数: {}, loss: {}".format(total_train_step, loss))
        # 现在已经有点样子了，但是更正规的写法其实loss的print会按下面的方法写
        print("训练次数: {}, loss: {}".format(total_train_step, loss.item()))
        # 二者的区别是，以一个tensor(5)的数据举例，如果是一个tensor数据类型，直接print会是tensor(5)，而加上item()后会变成一个数字5
        # 这里的字符串格式化自动转换为字符串了，所以看起来没差别
        # 加上item可以为后续的loss可视化作基础