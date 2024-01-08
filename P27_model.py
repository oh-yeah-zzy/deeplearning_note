import torch
from torch import nn

# 搭建神经网络
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
    
# 很多人喜欢在这里测试网络的正确性
# 那么怎么测试呢？
# 把网络模型创造出来
# 给网络模型一个确定的输入尺寸
# 看输出的尺寸是不是我们想要的
if __name__ == '__main__':
    tudui = Tudui()
    input = torch.ones((64, 3, 32, 32))
    output = tudui(input)
    print(output.shape)
    # input最前面的dim是batch_size是64，代表输入的是64张图片
    # output中后面的10代表每一张图片在10个类别中的概率是什么样子的，笔者认为这里输出的应该不是概率，因为还没有进行归一化