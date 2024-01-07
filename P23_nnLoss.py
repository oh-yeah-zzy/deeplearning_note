# 这节课先不讲pytorch给我们提供的一些网络模型
# 这节主要先讲解torch.nn中的loss function，来衡量误差
# loss function的作用是实际神经网络的输出和真实想要的结果中间的差距
# 神经网络训练的依据来源于loss function
# loss function一方面计算实际输出和目标之间的差距，另一方面为我们更新输出提供一定的依据(反向传播，是根据loss function和计算图来得到每个参数的更新大小)
# loss function使用起来不是很难，想要明白其中的计算方式需要有一定的数学功底
# 以nn.L1Loss为例，输入是x，目标是y，通过x和y进行相减后取绝对值，如果有多个则取平均
# 实际过程中要写代码的话还是要有些东西要注意
# 一定要注意输入输出的shape

import torch
from torch import nn
from torch.nn import L1Loss

# 在实际的数据或者网络中默认的tensor会是float数据类型，我们这里需要设置是因为我们创建tensor的时候没有加小数导致的，实际真实写代码一般是与不到的
inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss(reduction='sum')
result = loss(inputs, targets)

print(result)

# loss function看公式也许很绕，但是一定要关注输入输出的shape
# 再来看几个其他的loss function

loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs, targets)
print(result_mse)

# 这节再来讲一个交叉熵
# 这个比较复杂，一般用在训练分类问题
# 这个比较绕，注意好好看官方文档和例子

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3)) # (N, C)
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)

# 注意文档中的计算公式的log其实是以e为底的，也就是ln，而计算机的计算公式的log是以10为底的
# 一些文档中公式中的参数如果没有设置直接从公式中去掉就好了

