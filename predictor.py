



# 需要读取txt的一行并得到下面两个名字
import torch

filename = ''
framename = ''


# 这一段是我的预测部分，我会根据filename和framename来获取data并输入模型得到这一行的pred
def pred(filename, framename):
    # model = Model().cuda()
    # # print(model)
    # seqimgs = torch.rand(2, 2, 3, 256, 256).cuda()
    # vlen = torch.tensor([2, 2]).cuda()
    # poses = torch.rand(2, 2, 18, 2).cuda()
    # pred = model(seqimgs, vlen, poses)
    # print(pred)
    pred = torch.Tensor([[0.1015, 0.0741, 0.0924, 0.1710, 0.2129, 0.0871, 0.0987, 0.1622]])
    class_num = torch.argmax(pred, dim=1)  # 这里先给你一个预测值
    return int(class_num)


# 你调用pred就可以得到这一行的类别
classnum = pred(filename, framename)
print(classnum)
# 需要将类别填入txt文件对应帧的那一行
