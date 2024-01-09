# pytorch给我们提供了另一种采用GPU训练的方式
# .to方式
# 可以对网络模型、数据、损失函数调用.to(device)
# device是Device = torch.device("cpu")，这个代表device是cpu
# 也可以调用torch.deviec("cuda")，指定到cuda上面去
# 如果电脑上有一块显卡的话可以这么写，指定第一个显卡，torch.device("cuda:0")
# 如果电脑上有多块显卡，想指定第二块显卡的话可以这么写，torch.device("cuda:1")
# 对于单个显卡来说，torch.device("cuda")和torch.device("cuda:0")没有任何区别
# 接下来以代码为例
# 这种方式是我们更加常用的方式
# 其中还有一些细节
# 比如说对于网络模型，不需要另外赋值，直接调用.to就可以了，不需要重新赋值，.cuda也是如此，损失函数也是如此，不需要另外赋值
# 只有我们的数据、图片、标注需要另外转移后再给变量重新赋值
# 但是为了方便记忆，都可以直接赋值过去没有问题，但是看到别人那么写不要怀疑他有错
# 另外还有一个细节就是开头定义训练的设备，device
# device的话有的人喜欢这么写 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 这是一种语法糖，意思是如果cuda可以用就用GPU，如果cuda不可以用就用cpu
# 这个写法在代码中很常见，这也是为了适应各种环境，这是更加常用的写法
# 方式二是我们更加常用的训练方式

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

# 定义训练的设备，可以通过下面的变量控制在cpu上运行还是在gpu上运行
device = torch.device("cuda:0")
print(device)

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
# 调用模型.to方法
tudui = tudui.to(device)

loss_fn = nn.CrossEntropyLoss()
# 损失函数.to方法
loss_fn = loss_fn.to(device)

learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

total_train_step = 0
total_test_step = 0
epoch = 10

writer = SummaryWriter("./logs_train")
start_time = time.time()

for i in range(epoch):
    print("------------第 {} 轮训练开始------------".format(i+1))
    
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        
        # 训练数据.to()方法
        imgs = imgs.to(device)
        targets = targets.to(device)
        
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
            
            # 测试数据.to方法
            imgs = imgs.to(device)
            targets = targets.to(device)
            
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