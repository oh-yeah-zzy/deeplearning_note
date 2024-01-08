# 这节课讲解完整的模型训练套路
# 以CIFAR10作为这节课的例子，来完成对这个数据集的分类问题
# 这个数据集总共有10个类别，对于网络来说，这是一个10分类的问题
# 如果觉得数据集太简单，会在最后的时候来看一下github上优秀的代码或项目
# 模型训练套路讲完会讲GPU训练，然后再讲完整的模型验证套路

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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

# 添加tensorboard
writer = SummaryWriter("./logs_train")

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
        # 现在已经有点样子了，但是更正规的写法其实loss的print会按下面的方法写，说的是下面的print语句，关于为什么要加if，其实是为了防止输出信息太多导致看不到部分输出信息，这样可以避免很多的无用的信息
        if total_train_step % 100 == 0:
            print("训练次数: {}, loss: {}".format(total_train_step, loss.item()))
            # 逢百写入tensorboard
            writer.add_scalar("train_loss", loss.item(), total_train_step)

        # 二者的区别是，以一个tensor(5)的数据举例，如果是一个tensor数据类型，直接print会是tensor(5)，而加上item()后会变成一个数字5
        # 这里的字符串格式化自动转换为字符串了，所以看起来没差别
        # 加上item可以为后续的loss可视化作基础
        
        # 现在训练的步骤基本写完了
        # 那么怎么知道模型在训练的时候有没有训练好或者有没有达到需求呢？
        # 所以每次在训练完一轮之后进行测试
        # 以在测试集上的损失或者说在测试集上的正确率来评估模型有没有训练好
    
    # 要注意在测试的过程中不需要对模型进行调优了
    # 就是利用现有的模型进行测试
    # 所以要加下面的这句语句
    # 这句话的意思是说在with这个里面的代码没有了梯度
    # 相当于保证不会对网络进行调优
    # 所以就可以在里面写测试步骤
    # 测试步骤开始
    total_test_loss = 0
    # 计算正确率用total_accuracy
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            # 现在不是直接print loss
            # 我们在测试过程中想看的一般是我们训练的网络模型在整体测试数据集上的误差，或者说叫正确率
            # 我们现在求出来的loss只不过是一部分数据在网络模型上的损失
            # 但是我们想求的是整个数据集上的loss
            # 所以要设置一个total_test_loss的变量
            total_test_loss = total_test_loss + loss.item()
            # 计算每一次一小部分数据正确的个数，注意，最外层的中括号是第0维，所以这里传的是1
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))
    # 由于存在输出信息太多可能导致掩盖的问题，很多人其实喜欢这样子写，就是会修改前面输出训练次数的print语句，让他不会每一次都打印出来
    # 到这里我们的输出已经非常明朗了，但是我们仍可以进行一个优化，比如说我们可以加上之前说的tensorboard
    # 我们可以把每次的数据加到tensorboard中，他就可以画图出来
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1
    
    # tensorboard中的scalar曲线可能会有两条线，深色线是平滑处理后的线，浅色线是真实曲线
    # 到这里我们其实写的差不多了，但是我们忘记了比如说想保存每一轮训练的模型
    torch.save(tudui, "tudui_{}.pth".format(i))
    print("模型已保存")
    # 这个我们大概讲解完了
    # 还有一些优化，我们发现即使我们得到整体测试集上的loss，他好像也不能很好地说明我们的模型在测试集上表现的效果
    # 其实在分类问题中我们可以用正确率去进行表示，这部分内容可能会有点小困难，简单听一下就可以
    # 因为这是分类问题中常用的方式，如果说是目标检测或者语义分割的时候，其实最简单的方法是把得到的输出直接在tensorboard中显示
    # 所以后面要讲的正确率其实是分类问题中的一个比较特有的衡量指标
    
    # 正确率就是predict的label和真实的label相同的个数除以总个数
    # torch.tensor.argmax()能够求tensor顺着某一维度的最大值所在的下标
    # torch.tensor.sum()能够求tensor中的数据相加
    
writer.close()

# 这个就是训练的一个流程，这些循环中的细节还是很多的