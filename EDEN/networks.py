#author:Easo
#date:2019年4月15日

import torch
import torchvision as tv
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from collections import OrderedDict


class AutoNet(nn.Module):
    '''
    从染色体中自动构建网络的类
    '''
    def __init__(self,popMember,inputSize,inputChannel):
        '''
        网络初始化
        argvs:
            popMember:每个个体
            inputSize:输入图片的Size
        '''
        self.layerNums = popMember[0] + 1
        self.netMember = popMember[1]
        self.dataSize = inputSize
        self.convNums = 0
        self.fcNums = 0
        self.dropOutNums = 0
        self.poolNums = 0
        self.conv = None
        self.fc = None
        convList = []
        fc = []
        actFunc = {'liner':nn.Linear,
                    'leaky relu':nn.LeakyReLU(),
                    'prelu':nn.PReLU(),
                    'relu':nn.ReLU(),
                    'sigmoid':nn.Sigmoid(),
                    'softmax':nn.Softmax()
        }
        for layer in self.netMember:
            if layer[0] == 0:   #如果是卷积层
                self.convNums += 1
                convList.append(('conv'+str(self.convNums),
                                  nn.Conv2d(inputChannel,layer[3],layer[4],stride=1)
                ))
                padTmp = int((layer[4]-1)/2)
                self.dataSize = caculateSize(self.dataSize,layer[4],padTmp,1)   #计算一次datasize
                if layer[2] is not None:  #池化层
                    self.poolNums += 1
                    convList.append(('maxpool'+str(self.poolNums),
                                    nn.MaxPool2d(layer[2],stride=layer[2])
                    ))
                    self.dataSize = caculateSize(self.dataSize,layer[2],0,layer[2])
                if layer[1] is not None: #如果是dropout层
                    for do in layer[1]:
                        self.dropOutNums += 1
                    

def caculateSize(inputsize,kernelSize,padding,stride):
    '''
    计算size
    argvs:
        kernelSize:核的大小
        padding:填充
        stride:步长
    '''
    assert padding == int((kernelSize-1)/2)
    outputSize = int((inputsize + 2*padding - kernelSize)/stride) + 1
    return outputSize


if __name__ == "__main__":
    a = caculateSize(28,5,2,1)
    pass
    
