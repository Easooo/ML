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
        super(AutoNet,self).__init__()
        self.layerIndex = 1000
        self.netMember = popMember[1]
        self.dataSize = inputSize
        self.convNums = 0
        self.fcNums = 0
        self.dropOutNums = 0
        self.poolNums = 0
        self.conv = None
        self.fc = None
        self.featureMap = inputChannel
        self.featureMapAfterConv = 0
        self.midFcFlag = False
        convList = []
        fcList = []
        actFunc = {'linear':None,    #linear即激活函数为空，保持原有的线性结构
                    'leaky relu':nn.LeakyReLU(inplace=True),
                    'prelu':nn.PReLU(),
                    'relu':nn.ReLU(inplace=True),
                    'sigmoid':nn.Sigmoid(),
                    'softmax':nn.LogSoftmax(dim=1)
        }
        for layer in self.netMember:
            if layer[0] == 0:   
                ##########如果是卷积层###########
                self.convNums += 1
                padTmp = int((layer[4])/2)
                #k=1,p=0; k=2,p=1; k=3,p=1; k=4,p=2; k=5,p=2; k=6,p=3
                convList.append(('net '+str(self.layerIndex),
                                nn.Conv2d(self.featureMap,out_channels=int(layer[3]),kernel_size=layer[4],padding=padTmp)
                ))
                self.layerIndex += 1
                convList.append(('net'+str(self.layerIndex),
                                nn.BatchNorm2d(int(layer[3]))
                ))
                self.layerIndex += 1
                self.dataSize = caculateSize(self.dataSize,layer[4],padTmp,1)   #计算一次datasize    
                self.featureMap = layer[3]
                self.featureMapAfterConv = self.featureMap
                if layer[2] is not None:  #池化层
                    self.poolNums += 1
                    poolKernel = int(layer[2]) if self.dataSize > int(layer[2]) else int(self.dataSize)
                    #在卷积和池化操作中，可能会出现inputsize小于kernelsize的情况，
                    #在这种情况下，直接强制设置输出的size为1（在种群中被淘汰）
                    convList.append(('net '+str(self.layerIndex),
                                     nn.MaxPool2d(poolKernel,stride=int(layer[2]))
                    ))
                    self.layerIndex += 1
                    self.dataSize = caculateSize(self.dataSize,poolKernel,0,layer[2])
                #添加激活层
                if layer[-1] != 'linear':
                    convList.append(('net '+str(self.layerIndex),
                                     actFunc[layer[-1]]
                    ))
                    self.layerIndex += 1
                if layer[1] is not None: #如果是dropout层
                    for do in layer[1]:
                        self.dropOutNums += 1
                        convList.append(('net '+str(self.layerIndex),
                                        nn.Dropout(do)
                        ))
                        self.layerIndex += 1  
            elif layer[0] == 1: 
                #########如果是全连接层############
                self.midFcFlag = True
                self.fcNums += 1
                if self.fcNums == 1:
                    fcList.append(('net '+str(self.layerIndex),
                                    nn.Linear(int(self.featureMapAfterConv*self.dataSize*self.dataSize),
                                              int(layer[2])
                                    )
                    ))
                    self.layerIndex += 1
                    self.featureMap = layer[2]
                else:
                    fcList.append(('net '+str(self.layerIndex),
                                    nn.Linear(int(self.featureMap),int(layer[2]))
                    ))
                    self.layerIndex += 1
                    self.featureMap = layer[2]
                #添加激活层
                if layer[-1] != 'linear':
                    fcList.append(('net '+str(self.layerIndex),
                                     actFunc[layer[-1]]
                    ))
                    self.layerIndex += 1
                if layer[1] is not None: #如果是dropout层
                    for do in layer[1]:
                        fcList.append(('net '+str(self.layerIndex),
                                        nn.Dropout(do)
                        ))
                        self.layerIndex += 1 
                        self.dropOutNums += 1
            else:#最后一层fc
                self.fcNums += 1
                if self.midFcFlag:   #如果有中间FC，输入的神经元数为上一层fc的输出神经元数
                    fcList.append(('net '+str(self.layerIndex),
                                    nn.Linear(int(self.featureMap),int(layer[1]))
                    ))
                    self.layerIndex += 1
                else:
                    fcList.append(('net '+str(self.layerIndex),
                                    nn.Linear(int(self.featureMapAfterConv*self.dataSize*self.dataSize),int(layer[1]))
                    ))
                    self.layerIndex += 1
                if layer[-1] != 'linear':                    
                    fcList.append(('net '+str(self.layerIndex),
                                    actFunc[layer[-1]]
                    ))
                    self.layerIndex += 1
        self.conv = nn.Sequential(OrderedDict(convList))
        self.fc = nn.Sequential(OrderedDict(fcList))

    def forward(self,inputData):
        #定义前向过程
        convRes = self.conv(inputData)
        convOnedim = convRes.view(-1,self.dataSize*self.dataSize*self.featureMapAfterConv)
        res = self.fc(convOnedim)
        return res



                
def caculateSize(inputsize,kernelSize,padding,stride):
    '''
    计算size
    argvs:
        kernelSize:核的大小
        padding:填充
        stride:步长
    '''
    outputSize = int((inputsize + 2*padding - kernelSize)/stride) + 1
    return outputSize

    
