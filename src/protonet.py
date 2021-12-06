import torch.nn as nn
import torchvision
#from __future__ import print_function  # 这个是python当中让print都以python3的形式进行print，即把print视为函数
import argparse  # 使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
import torch  # 以下这几行导入相关的pytorch包，有疑问的参考我写的 Pytorch打怪路
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchnet import meter

'''
def conv_block(in_channels, out_channels):
   # returns a block conv-bn-relu-pool
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
'''

class ProtoNet(nn.Module):
    
    #Model as described in the reference paper,
    #source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    
    def __init__(self):
        super(ProtoNet, self).__init__()
        '''
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        '''
        self.conv = nn.Conv2d(1, 3, 1)
        self.resnet = torchvision.models.mobilenet_v2(pretrained=False)

    def forward(self, x):
        #x = self.encoder(x)
        x = self.conv(x)
        x = self.resnet(x)
        return x.view(x.size(0), -1)
