# 可以非常直观地看到每个步骤给model提供了哪些数据
# 或者可以看到model每个step的输出结果
# 或者可以观察不同阶段的一些显示
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
image_path = "dataset/train/ants/0013035.jpg"
image_PIL = Image.open(image_path)
image_array = np.array(image_PIL)
print(image_array.shape)

# 从PIL到numpy，需要在add_image()中指定shape中每一个数字/维表示的含义，即后面的dataformats
# 如果title不变，step和图像变了，则会在同一个title下出现滑块
# 如果想要换title，则可以设定不同的tag
writer.add_image("test", image_array, 1, dataformats='HWC') # 常用来观察训练结果

# y = x
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)

writer.close()
