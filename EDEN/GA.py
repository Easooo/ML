#author:Easo
#date:2019年4月12日

import matplotlib.pyplot as plt
import numpy as np

"""
The GA of Evo Net
"""
class Genetic(object):
    '''
    GA算法类,包括变异,选择,交叉(待做)
    '''
    def __init__(self):
        self.populationSize = None
        self.layerType = ['conv','fc','dropout']
        self.actTypeForConv = ['liner','leaky relu','prelu','relu']
        self.actTypeForFc = ['liner','sigmoid','softmax','relu']  #注如果是最后一层FC的话是没有relu的
        self.learningRate = np.arange(1e-4,1e-3,1e-5)
        self.nHiddenLayers = np.arange(2,8)  #隐含层的范围为2-7,即神经网络层数范围为4-9(算上input和output)
        self.fcUnits = np.arange(10,101)   #10 - 100,这个fc是隐含层中的fc，不是最后的fc
        self.convFilterNum = np.arange(10,101)
        self.kernelSize = np.arange(1,7)
        self.keepProb = np.arange(0.3,1.05,0.05)   #keep prob for dropout


    def createPopulation(self,populationSize):
        '''
        创建种群并初始化，返回一个包含种群的list
        returns:
            一个包含种群的list,在该list中,list[1]为net结构,对于net[0]:
            0表示卷积层
            1表示fc层
            2表示最后一层fc层
            在net结构里,如果是conv,则net[1]和net[2]分别为pool,droptimes
            conv:[0,[dropout],maxpool_ksize,fliterNums,kerSize,actType]
            fc:[1,[dropout],unitsNums,actType]
            lastFc:[2,unitsNum=classify,actType]

        '''
        self.populationSize = populationSize
        popMember = []
        for popSize in range(self.populationSize):
            chromosome = []
            net = []   
            #net:net由多个代表layer的list组成,layer数不等于hidenlayernums,因为do层写进了conv或fc里面
            hiidenLayerNums = int(popSize/10) + 1
            # hiidenLayerNums = 5
            print('隐含层数量:',hiidenLayerNums)
            chromosome.append(hiidenLayerNums)
            #=====================================================|
            #                                                     |
            # chromosome: [hiddenLayernums,net,lr,(acc,paraNums)] |
            #                                                     |
            #=====================================================|
            cnnFlag = True
            i = 0
            while (i < hiidenLayerNums):
                if i == 0:
                    dropoutAllow = False
                else:
                    dropoutAllow = True
                dropFlag,maxpoolflag,newLayer = self.createLayers(cnnFlag,dropoutAllow)
                if dropFlag:  #如果有dropout操作
                    prelayer = net[-1]  #当前net的最后一层,添加do操作
                    if prelayer[1] is None:
                        prelayer[1] = []
                    prelayer[1].append(newLayer)  #实际上是放入keepprob参数
                    i += 1
                elif maxpoolflag:
                    net.append(newLayer)
                    i += 2     #卷积+池化 两层
                else:                                  
                    net.append(newLayer)
                    i += 1
                    if newLayer[0] == 1:  #fc层
                        cnnFlag = False
            #fc创建最后一层fc
            subNet = [2,10]   #2代表last fc,10代表units数量
            subNet.append(np.random.choice(self.actTypeForFc[:-1]))  #最后一层fc不需要relu
            net.append(subNet)
            chromosome.append(net)
            lr = np.random.rand()/100   #随机初始化学习率
            chromosome.append(lr)
            acc,netParaNums= 0,0    #准确率和网络参数数量
            chromosome.append((acc,netParaNums))
            popMember.append(chromosome)
        return popMember       


    def createLayers(self,cnnFlag,dropoutAllow):
        '''
        创建网络层
        argvs:
            cnnFlag:设置下一层网络是否可以是cnn
            dropoutAllow:设置是否可以使用dropout
        return:
            一个关于网络结构的元组(dropflag,list),dropflag = 1表示是dropout
        '''
        dropFlag = 0
        maxpoolflag = 0
        if cnnFlag:
            if dropoutAllow:
                layerTypeTmp = np.random.choice(self.layerType)
                if layerTypeTmp == 'conv':
                    layer = createConv(self.convFilterNum,self.kernelSize,self.actTypeForConv)
                    poolkSize = createMaxpool(self.kernelSize)   #创建池化层
                    if poolkSize is not None:
                        layer[2] = poolkSize           #将池化层参数插入layer中
                        maxpoolflag = 1
                    return (dropFlag,maxpoolflag,layer)
                if layerTypeTmp == 'fc':
                    layer = createFc(self.fcUnits,self.actTypeForFc)
                    return (dropFlag,maxpoolflag,layer)
                if layerTypeTmp == 'dropout':
                    dropFlag = 1
                    prob = np.random.choice(self.keepProb)
                    return (dropFlag,maxpoolflag,prob)                   
            else:
                layer = createConv(self.convFilterNum,self.kernelSize,self.actTypeForConv)
                poolkSize = createMaxpool(self.kernelSize)   #创建池化层
                if poolkSize is not None:
                    layer[2] = poolkSize           #将池化层参数插入layer中
                    maxpoolflag = 1               
                return (dropFlag,maxpoolflag,layer) 
        else:   #这里不能用cnn 因为是fc
            if dropoutAllow:
                layerTypeTmp = np.random.choice(self.layerType[1:])
                if layerTypeTmp == 'fc':
                    layer = createFc(self.fcUnits,self.actTypeForFc)
                    return (dropFlag,maxpoolflag,layer)
                if layerTypeTmp == 'dropout':
                    dropFlag = 1
                    prob = np.random.choice(self.keepProb)
                    return (dropFlag,maxpoolflag,prob) 
            else:
                layer = createFc(self.fcUnits,self.actTypeForFc)
                return (dropFlag,maxpoolflag,layer)

 
    def mutate(self):
        pass

    def selcet(self):
        pass

    def getFitness(self):
        pass

def getChoiceBool(choiceRate=0.5):
    '''
    随机选择
    argvs:
        choiceRate:选择的概率,0~1
    returns:
        bool类型，选中为True
    '''
    rand = np.random.choice(range(0,101))
    boolFlag = (rand <= choiceRate*100)
    return boolFlag

def createConv(convFilterNum,kernelSize,actTypeForConv):
    '''
    创建卷积层,返回layer
    layer[0] = 0代表卷积层,layer[1]=None是初始化设计,后续会根据dropflag调整
    layer[2]是池化
    '''
    layer = [0,None,None,0,0,None]   

    FilterNum = np.random.choice(convFilterNum)  
    conkSize = np.random.choice(kernelSize)
    layer[3] = FilterNum
    layer[4] = conkSize
    actType = np.random.choice(actTypeForConv)
    layer[-1] = actType
    return layer

def createMaxpool(kernelSize):
    '''
    随机创建池化层
    '''
    poolkSize = None
    if getChoiceBool(0.5):
        poolkSize = np.random.choice(kernelSize)
    return poolkSize


def createFc(fcUnits,actTypeForFc):
    '''
    创建全连接层,返回layer
    '''
    layer = [1,None,0,None]
    units = np.random.choice(fcUnits)
    layer[2] = units
    actType = np.random.choice(actTypeForFc)
    layer[-1] = actType
    return layer


if __name__ == '__main__':
    test = Genetic()
    pop = test.createPopulation(100)
    pass