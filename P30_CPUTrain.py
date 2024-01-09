# 使用GPU训练在原来的代码中仅需修改几处即可
# 使用GPU训练有两种方式，可以用两种方式实现代码在GPU上进行训练

# 第一种GPU训练的方式
# 找到网络模型，找到数据（输入、标注），找到损失函数，然后调用他们的.cuda()，然后进行返回就可以
# 把之前P29的代码全选复制过来，这里把之前代码的注释全部删掉了，因为实在太多了
# 为了后面演示方便，这里把模型注释掉，然后把真实的model拷贝过来
# 现在这个文件是一个完整的文件了，不需要从外部引入模型

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

train_data = torchvision.datasets.CIFAR10(root="./data", train=True, transform=torchvision.transforms.ToTensor(),
                                         download=True)
test_data = torchvision.datasets.CIFAR10(root="./data", train=False, transform=torchvision.transforms.ToTensor(),
                                          download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集的长度为: {}".format(train_data_size))
print("测试数据集的长度为: {}".format(test_data_size))

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )
        
    def forward(self, x):
        x = self.model(x)
        return x
tudui = Tudui()

loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

total_train_step = 0
total_test_step = 0
epoch = 10

writer = SummaryWriter("./logs_train")
# 记录下当前时间
start_time = time.time()

for i in range(epoch):
    print("------------第 {} 轮训练开始------------".format(i+1))
    
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数: {}, loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
            
    print("整体测试集上的loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1
    
    torch.save(tudui, "tudui_{}.pth".format(i))
    print("模型已保存")
    
writer.close()