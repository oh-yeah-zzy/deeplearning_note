# 这节我们讲完整的模型验证套路，有的时候我们也管他叫测试或者demo
# 核心就是利用已经训练好的模型给他提供输入
# 和之前模型训练套路中的测试部分其实是差不多的
# 如果模型训练好了，这个模型就是可以对外提供的一个实际的应用

import torch
import torchvision
from PIL import Image
from torch import nn

device = torch.device("cuda:0")
image_path = "./imgs/dog.png"
image = Image.open(image_path)
print(image)
image = image.convert('RGB')
print(image)

# 完成图像的加载后，我们发现我们图像的输入和网络的输入不匹配，图像的输入大于网络要求的图像输入大小，所以我们要进行resize
# 如果我们的图像是png格式，则还需要加image = image.convert('RGB')
# 因为png图像是四个通道，除了RGB三通道外，还有一个透明度通道，所以调用image = image.convert('RGB')，保留其颜色通道
# 若图片本来就是三个颜色通道，则经过此操作图像不变
# 加上这一步后，可以适应png、jpg等各种格式的图片
# 需要注意不同截图软件截下来的图片保留的通道数可能是不一样的

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)

# 可以看到图片的大小已经被resize成3 * 32 * 32了
# 下一步就可以加载网络模型了
# 加载网络模型的过程中，因为之前是采用第一种方式保存的，所以要采用第一种方式加载

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

# 接下来加载网络模型
# 如果用的是GPU上训练的模型，然后在一台只有cpu的机器上面跑，加载的时候需要设置map_location字段
model = torch.load("tudui_0.pth")
print(model)
model = model.to(device)

# 这部分容易被遗忘，因为训练过程中是需要batch_size的
image = torch.reshape(image, (1, 3, 32, 32))
print(image.shape)
image = image.to(device)

# 下面这两步容易遗忘
model.eval()
# 下面这一步能够节约一些内存和性能
with torch.no_grad():
    # 注意这里的.cuda，因为模型保存的时候是用GPU训练的，所以这里也需要定义device到cuda
    # 否则会报错RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same or input should be a MKLDNN tensor and weight is a dense tensor
    output = model(image)
    
print(output)
# 要把无效输出转换成利于解读的方式
print(output.argmax(1))
# 对应的类别可以在使用dataset加载数据集后debug进行查看
# 这里没猜中的主要原因是模型没怎么训练

# test主要就是把我们训练的模型应用到我们的实际环境中