# torchvision.dataset 会提供一些数据集
# torchvision.io 不常用，处理输入输出用的
# torchvision.model 里面有一些比较常用的神经网络，有的已经预训练好了，比较重要，后面会讲到
# torchvision.ops torchvision 提供的一些特殊操作，基本是用不到的
# torchvision.transforms 之前讲解过了
# torchvision.utils 一些常用的小工具，tensorboard就来自于这个模块
# 这一次主要讲解transforms进行一个联合的使用
# 这次主要讲解torchvision.dataset以及dataset如何与transform进行一个联合的使用
# 本次代码中的target不是标签，是标签的位置，标签列表是classes，真正的标签是classes[target]

import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False,transform=dataset_transform, download=True)

# print(test_set[0])
# print(test_set.classes)

# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

# print(test_set[0])

writer = SummaryWriter("p10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()
