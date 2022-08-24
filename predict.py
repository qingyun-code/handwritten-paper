import torch

# pretreatment.py为上面图片预处理的文件名，导入图片预处理文件
import pretreatment as PRE

# 加载网络模型
net = torch.load('weight/test.pkl')

# 得到返回的待预测图片值，就是pretreatment.py中的zoom_image
img = PRE.image_preprocessing()

# 将待预测图片转换形状
inputs = img.reshape(-1, 784)

# 输入数据转换成tensor张量类型，并转换成浮点类型
inputs = torch.from_numpy(inputs)
inputs = inputs.float()

# 丢入网络进行预测，得到预测数据
predict = net(inputs)

# 打印对应的最后的预测结果
print("The number in this picture is {}".format(torch.argmax(predict).detach().numpy()))