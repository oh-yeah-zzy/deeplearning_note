# Dataset就是一个数据集
# Dataloader是一个加载器，就是把数据加载到神经网络中，他所做的事就是每次从dataset中取数据，每次取多少怎么取是由Dataloader的参数进行设置的
# Dataloader的参数比较多，但是只有一个dataset参数没有默认值，其实就是之前学的自定义的dataset，dataset就是告诉程序数据在什么地方，只需要把之前做的dataset实例化然后放到这里就可以了，其他的参数有默认值，一般只需要设置少量的就可以
# 这节讲解一些常用的参数设置
# batch_size 每次放多少笔数据进神经网络
# shuffle 是否打乱的意思，默认设为False，但是我们喜欢设置为True，注意这里遍历完一遍后才会重新打乱，相当于把牌全摸完了重新洗牌
# sampler和batch_sampler后面进行讲解
# num_workers 多进程，加载数据的时候是采用单个进程还是多个进程进行加载，多个进程的速度比较快，默认设置为0，0的话是采用主进程进行加载，但是这个参数在windows底下会有一些问题
# drop_last 数据集总数除以batch_size后除不尽的多余的数据是否舍去

import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# 测试数据集中第一张图片及target
img, target = test_data[0]
print(img.shape)
print(target)

# 外面这一层循环是为了讲解shuffle的作用
for epoch in range(2):
    # 里面这一层循环是为了讲解drop_last和batch_size的作用
    writer = SummaryWriter("dataloader")
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step = step + 1

writer.close()

# test_data中的第一张和test_loader中的第一张可能会不一样，因为我们的sampler没有进行任何的设置，有可能是torch.utils.data.sampler.RandomSampler，sampler规定了从test_data中抓取batch的策略
# tensorboard在数据比较多的时候会筛选显示，可以在终端后面加上一句"--samples_per_plugin images=x"，x是想要显示的步数的数量，越多越好
# 在实际的训练过程中我们也是这么写的，就是for data in test_loader:这样子，然后把imgs输入到神经网络中去
