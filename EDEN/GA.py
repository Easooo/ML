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
        self.layerType = ['conv','maxpool','fc']
        self.actTypeForConv = ['liner','leaky relu','prelu','relu']
        self.actTypeForFc = ['liner','sigmoid','softmax','relu']  #注如果是最后一层FC的话是没有relu的
        self.learningRate = np.arange(1e-5,1e-4)
        self.nHiddenLayers = np.arange(2,8)  #隐含层的范围为2-7,即神经网络层数范围为4-9(算上input和output)
        self.fcUnits = np.arange(10,101)   #10 - 100,这个fc是隐含层中的fc，不是最后的fc
        self.convFilterNum = np.arange(10,101)
        self.kernelSize = np.arange(1,7)
        self.keepProb = np.arange(0.5,1.05,0.05)   #keep prob for dropout
    def createPopulation(self,populationSize):
        '''
        创建种群并初始化，返回一个包含种群的list
        returns:
            一个包含种群的list,在该list中,list[1]为net结构,对于net[0]:
            0表示卷积层
            1表示池化层
            2表示dropou层
            3表示fc层,4表示最后一层fc层

        '''
        self.populationSize = populationSize
        popMember = []
        for _ in range(self.populationSize):
            chromosome = []
            net = []
            hiidenLayerNums = np.random.choice(self.nHiddenLayers)
            chromosome.append(hiidenLayerNums)
            cnnFlag = True
            for i in range(hiidenLayerNums):
                if i == 0:
                    dropoutFlag = False
                else:
                    dropoutFlag = True
                newLayer = self.createLayers(cnnFlag,dropoutFlag)
                net.append(newLayer)
                if net[0] == 3:  #fc层
                    cnnFlag = False
            #fc创建最后一层fc
            subNet = [4,10]   #4代表last fc,10代表units数量
            subNet.append(np.random.choice(self.actTypeForFc[:-1]))  #最后一层fc不需要relu
            net.append(subNet)
            chromosome.append(net)
            lr = np.random.choice(self.learningRate)   #随机初始化学习率
            chromosome.append(lr)
            acc,netParaNums= 0,0    #准确率和网络参数数量
            chromosome.append((acc,netParaNums))
            popMember.append(chromosome)
                          
    def createLayers(self,cnnFlag,dropoutFlag):

        pass
 

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

if __name__ == '__main__':
    a = getChoiceBool(0.5)
    pass