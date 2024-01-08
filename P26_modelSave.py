# 目前已经基本上讲完架构了
# 优化器那节已经在训练模型了
# 但是模型的保存和加载目前还没有说
# 这两个20多分钟就讲完了
# 接下来会讲解完整的模型训练套路，这里也会讲一个GPU训练
# 再接下来讲一个完整的模型验证套路
# 如果有时间的话再来看一下github上常见的一些代码，其中会有一些陷阱，对新手不友好，没有接触的话有些参数会不是很了解
# 模型训练套路的话我们要注意从一开始就要养成优秀的写法
# 或者说去学习一个优秀的套路模板
# 那这个模板从哪里来呢？最推荐的就是看pytorch官网，他给我们推荐的教程或者提供的一种程序
# 这节会以pytorch官网的教程为模板进行讲解
# 相当于把模板拿过来改动一下，其中的步骤和套路就按他的来

import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(weights=None)

# 第一种保存方式，保存方式一
torch.save(vgg16, "vgg16_method1.pth")
# 这种方式一保存的时候不仅保存了网络模型的结构，也保存了网络模型中的参数

# 保存方式二（官方推荐的保存方式）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")
# 这种方式二相当于把vgg16的状态保存成字典形式，vgg16网络模型的参数保存成字典，相当于不再保存网络模型的结构，只保存网络模型的参数
# 通过这种形式可以把vgg16的参数保存成python中的字典形式

# 第二种保存方式占用的硬盘空间比较小，windwos下用dir命令能够看到所占的空间

#保存方式一其实是有陷阱的
class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        
    def forward(self, x):
        x = self.conv1(x)
        return x
    
tudui = Tudui()
torch.save(tudui, "tudui_method1.pth")
# 用现有的网络模型是看不出这个陷阱的，但是用自己的网络模型就可以看出来这个陷阱