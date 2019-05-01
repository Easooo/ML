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
        self.maxlayerNums = -1
        self.layerType = ['conv','fc','dropout']
        self.actTypeForConv = ['linear','leaky relu','prelu','relu']
        self.actTypeForFc = ['linear','sigmoid','softmax','relu']  #注如果是最后一层FC的话是没有relu的
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
        popTotal = []
        for popSize in range(self.populationSize):
            chromosome = []
            net = []   
            #net:net由多个代表layer的list组成,layer数不等于hidenlayernums,因为do层写进了conv或fc里面
            self.maxlayerNums = int(popSize/10) + 1
            # hiidenLayerNums = int(popSize/10) + 1
            # hiidenLayerNums = 5
            # print('隐含层数量:',self.maxlayerNums)
            chromosome.append(self.maxlayerNums)
    #=================================================================================|
    #                                                                                 |
    # chromosome: [hiddenLayernums,net,lr,[acc,paraNums],fitness,mutateFlag,popindex] |
    # 隐含层第一层必定是卷积层,fitness待初始化为0                                        |
    # chromosome[0]:隐含层数
    # chromosome[1]:网络结构
    # chromosome[2]:学习率
    # chromosome[3]:[0]:正确率 [1]:超参数数量
    # chromosome[4]:适应度
    # chromosome[5]:是否变异
    # chromosome[6]:索引,以便于演化到最后的时候容易超出是哪一个并且和模型对应
    #=================================================================================
            cnnFlag = True
            i = 0
            while (i < self.maxlayerNums):
                if i == 0:
                    dropoutAllow = False
                else:
                    dropoutAllow = True
                dropFlag,maxpoolflag,newLayer = self.createLayers(cnnFlag,dropoutAllow,i)
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
            acc,netParaNums = 0,0    #准确率和网络参数数量
            fitness = 0  #初始化适应度
            chromosome.append([acc,netParaNums])
            chromosome.append(fitness)
            chromosome.append(False)
            chromosome.append(popSize)
            popTotal.append(chromosome)
        return popTotal       


    def createLayers(self,cnnFlag,dropoutAllow,currentLayerNum):
        '''
        创建网络层
        argvs:
            cnnFlag:设置下一层网络是否可以是cnn
            dropoutAllow:设置是否可以使用dropout
            currentLayerNum:当前已创建的层,设置这个层是为了限制 当前层的数量和最大层数相差为1的时候，
            卷积层和池化层同时创建
        return:
            一个关于网络结构的元组(dropflag,list),dropflag = 1表示是dropout
        '''
        dropFlag = 0
        maxpoolflag = 0
        currentlayerNumTmp = currentLayerNum
        if cnnFlag:
            if dropoutAllow:
                layerTypeTmp = np.random.choice(self.layerType)
                if layerTypeTmp == 'conv':
                    if currentlayerNumTmp < self.maxlayerNums:
                        layer = createConv(self.convFilterNum,self.kernelSize,self.actTypeForConv)
                        currentlayerNumTmp +=1
                    if currentlayerNumTmp < self.maxlayerNums:
                        poolkSize = createMaxpool(self.kernelSize)   #创建池化层                        
                        if poolkSize is not None:
                            layer[2] = poolkSize           #将池化层参数插入layer中
                            currentlayerNumTmp +=1
                            maxpoolflag = 1
                    return (dropFlag,maxpoolflag,layer)
                if layerTypeTmp == 'fc':
                    if currentlayerNumTmp < self.maxlayerNums:
                        layer = createFc(self.fcUnits,self.actTypeForFc)
                        return (dropFlag,maxpoolflag,layer)
                if layerTypeTmp == 'dropout':
                    if currentlayerNumTmp < self.maxlayerNums:
                        dropFlag = 1
                        prob = np.random.choice(self.keepProb)
                        return (dropFlag,maxpoolflag,prob)                   
            else:
                if currentlayerNumTmp < self.maxlayerNums:
                    layer = createConv(self.convFilterNum,self.kernelSize,self.actTypeForConv)
                    currentlayerNumTmp += 1
                if currentlayerNumTmp < self.maxlayerNums:
                    poolkSize = createMaxpool(self.kernelSize)   #创建池化层
                    if poolkSize is not None:
                        layer[2] = poolkSize           #将池化层参数插入layer中
                        currentlayerNumTmp += 1
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

 
    def mutate(self,popMember,maxLayer=10):
        '''
        变异操作
        popmember:待变异的个体
        maxlayer:最大的隐含层数量,默认为10
        '''
        if getChoiceBool(0.1): #学习率
            newLr = np.random.rand()/100
            popMember[2] = newLr
            return 
        else:  #网络层结构的变异        
            opFuncDict = {'add':mutateAddLayer,'rep':mutateReplaceLayer,'del':mutateDelLayer} 
            #分别代表增加\删除\替换
            if popMember[0] == 1:   #只有一层,只能执行增加和替换功能
                operationType = ['add','rep']  #不能删除
                while True:                     #直到正确为止
                    operation = np.random.choice(operationType)
                    if operation == 'add':
                        layeridx = np.random.choice(range(popMember[0] + 1))#0、1随便选一个
                    else:
                        layeridx = popMember[0] - 1   #为0
                    opFunc = opFuncDict[operation]
                    layerType = np.random.choice(['conv','pool','drop','fc'])
                    cf,newLayer = opFunc(popMember[1],layeridx,layerType)
                    if cf:
                        break
                if operation == 'add':
                    popMember[0] += 1
                    popMember[1] = newLayer
                    popMember[5] = True  
                else:
                    popMember[1] = newLayer
                    popMember[5] = True

            elif 1 < popMember[0] and popMember[0] < maxLayer:  #大于一层小于最大层
                operationType = ['add','rep','del']  #不能删除
                while True:                     #直到正确为止
                    operation = np.random.choice(operationType)
                    if operation == 'add':
                        layeridx = np.random.choice(range(popMember[0]+1)) 
                        #可以加在倒数最后一层,因为insert方法会使得list后移
                    else:
                        layeridx = np.random.choice(range(popMember[0]))
                    opFunc = opFuncDict[operation]
                    layerType = np.random.choice(['conv','pool','drop','fc'])
                    cf,newLayer = opFunc(popMember[1],layeridx,layerType)
                    if cf:
                        break
                if operation == 'add':
                    popMember[0] += 1
                    popMember[1] = newLayer
                    popMember[5] = True

                elif operation == 'del':
                    popMember[0] -= 1
                    popMember[1] = newLayer
                    popMember[5] = True
                else:
                    popMember[1] = newLayer
                    popMember[5] = True                   


            else:   #等于最大层数
                operationType = ['rep','del']  #不能增加
                while True:                     #直到正确为止
                    operation = np.random.choice(operationType)
                    layeridx = np.random.choice(range(popMember[0]))
                    opFunc = opFuncDict[operation]
                    layerType = np.random.choice(['conv','pool','drop','fc'])
                    cf,newLayer = opFunc(popMember[1],layeridx,layerType)
                    if cf:
                        break
                if operation == 'del':
                    popMember[0] -= 1
                    popMember[1] = newLayer
                    popMember[5] = True
                else:
                    popMember[1] = newLayer
                    popMember[5] = True   

    def selcet(self,population,tournaSize=7):
        '''
        通过锦标赛规则选择父代
        argvs:
            population:种群
            tournaSize:参与锦标赛的数量
        return:
            最好的染色体(个体)和index
        '''
        populationSize = len(population)
        idx = np.random.choice(np.arange(populationSize),size=tournaSize,replace=False)
        populationNp = np.array(population)
        tournaPop = list(populationNp[idx])   #选择出来的一组锦标赛竞争者
        tournaPopidx = list(zip(tournaPop,idx))
        tournaPopSort = sorted(tournaPopidx,key=lambda popmember:popmember[0][2]) #fitness在第4
        #返回fitness最小的一个popmember和在原来的种群中的index
        return list(tournaPopSort[0][0]),int(tournaPopSort[0][1])  


    def getFitness(self,popMember):
        '''
        计算适应度,适应度的值越小越好
        '''
        AF = 1
        acc,netParaNums = popMember[3][0],popMember[3][1]
        fitness = (1 - acc) + AF*(1 - 1/netParaNums)
        popMember[4] = fitness   #修改fitness
        return fitness

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

def departLayers(netLayer):
    '''
    将population中的组合net分离成单个的列表形式
    e.g.:
        [[0, [0.5,0.3], 4, 49, 3, 'linear'], 
        [0, None, None, 18, 2, 'relu'], 
        [1, [0.8], 78, 'linear'], 
        [2, 10, 'softmax']]    最后一层不算,hiddenlayernums = 7
    ==>
        [['conv',out=49,kersize=3,'linear'],
         ['pool',kersize=4],
         ['drop',0.5],
         ['drop',0.3],
         ['conv',out=18,kersize=2,'relu'],
         ['fc',out=78,'linear'],
         ['drop',0.8]
         ['lastfc',10,'softmax']
        ]             
    '''
    departLayersList = []
    for everynet in netLayer:
        conv = ['conv']
        pool = ['pool']
        fc = ['fc']
        if everynet[0] == 0: #卷积层
            conv = conv + everynet[3:]
            departLayersList.append(conv)
            if everynet[2] is not None: #池化层
                pool.append(everynet[2])
                departLayersList.append(pool)
            if everynet[1] is not None:   #dropout层
                for kp in everynet[1]:
                    drop = ['drop'] + [kp]
                    departLayersList.append(drop)
        elif everynet[0] == 1: #全连接层
            fc = fc + everynet[2:]
            departLayersList.append(fc)
            if everynet[1] is not None: #dropout层
                for kp in everynet[1]:
                    drop = ['drop'] + [kp]
                    departLayersList.append(drop)
        else:
            lastfc = ['lastfc'] + everynet[1:]
            departLayersList.append(lastfc)
            return departLayersList    #已经到了最后一层fc

def mergeLayers(departLayersList):
    '''
    将单个的列表形式合并成population中的组合net
    e.g.:
        [['conv',out=49,kersize=3,'linear'],
         ['pool',kersize=4],
         ['drop',0.5],
         ['drop',0.3],
         ['conv',out=18,kersize=2,'relu'],
         ['fc',out=78,'linear'],
         ['drop',0.8]
         ['lastfc',10,'softmax']
        ]             #自动忽略最后一层全连接层
    ==>
        [[0, [0.5,0.3], 4, 49, 3, 'linear'], 
        [0, None, None, 18, 2, 'relu'], 
        [1, [0.8], 78, 'linear'], 
        [2, 10, 'softmax']]    最后一层不算,hiddenlayernums = 7 

    '''
    mergeLayersList = []
    LayerTmp = None
    convtotal = []
    fctotal = []
    fcflag = False
    for layers in departLayersList:
        if layers[0] == 'conv':  #卷积
            if LayerTmp is not None:
                mergeLayersList.append(LayerTmp)
                LayerTmp = None
                convtotal = []
            convtotal = [0,None,None] + layers[1:]
            LayerTmp = convtotal
        elif layers[0] == 'pool': #池化层
            convtotal[2] = layers[1]
            LayerTmp = convtotal
        elif layers[0] == 'drop' and (not fcflag):  #dropout层
            if convtotal[1] is None:
                convtotal[1] = [layers[1]]
                LayerTmp = convtotal
            else:
                convtotal[1].append(layers[1])
                LayerTmp = convtotal
        elif layers[0] == 'fc':   #全连接层
            fcflag = True
            if LayerTmp is not None:
                mergeLayersList.append(LayerTmp)
                LayerTmp = None
                fctotal = []
            fctotal = [1,None] + layers[1:]
            LayerTmp = fctotal
        elif layers[0] == 'drop' and (fcflag):
            if fctotal[1] is None:
                fctotal[1] = [layers[1]]
                LayerTmp = fctotal
            else:
                fctotal[1].append(layers[1])
                LayerTmp = fctotal
        elif layers[0] == 'lastfc':
            mergeLayersList.append(LayerTmp)
            LayerTmp = None
            LayerTmp = [2] + layers[1:]
            mergeLayersList.append(LayerTmp)
    return mergeLayersList

def mutateReplaceLayer(netList,layerIdx,layerType):
    '''
    替换操作,需要检查替换后的网络合理性
    argvs:
        netList:popmember[1]
        layerIdx:需要操作的index
        layerType:将要替换的层的类型
    '''
    assert layerType in ['conv','pool','drop','fc']
    netDepartTmp = departLayers(netList)
    # netDepart = netDepartTmp[:-1] #去除最后一层fc
    CFLAG = False
    newNetList = None
    if layerType == 'conv':
        if layerIdx != 0:   #layeridx = 0的时候可以继续换conv
            for i in netDepartTmp[:layerIdx]:    
                if i[0] == 'fc':     #卷积不能在fc后面
                    return CFLAG,newNetList
        FilterNum = np.random.choice(np.arange(10,101))  
        conkSize = np.random.choice(np.arange(1,7))
        actType = np.random.choice(['linear','leaky relu','prelu','relu'])
        newLayer = ['conv',FilterNum,conkSize,actType]
        netDepartTmp[layerIdx] = newLayer
        CFLAG = judgeConv(netDepartTmp)   #判断
        if CFLAG:
            newNetList = mergeLayers(netDepartTmp)
            return CFLAG,newNetList   #返回网络正确与否的flag,新的网络(格式与popmember一样)

    elif layerType == 'pool':  #池化层
        if layerIdx == 0:   #第0层直接pass
            return CFLAG,newNetList
        for i in netDepartTmp[:layerIdx]:
            if i[0] == 'fc':    #池化不能在fc后面
                return CFLAG,newNetList
        poolkSize = np.random.choice(np.arange(1,7))
        newLayer = ['pool',poolkSize]
        netDepartTmp[layerIdx] = newLayer  #替换
        CFLAG = judgeConv(netDepartTmp)   #判断
        if CFLAG:
            newNetList = mergeLayers(netDepartTmp)
            return CFLAG,newNetList   #返回网络正确与否的flag,新的网络(格式与popmember一样)

    elif layerType == 'drop':
        if layerIdx == 0:   #第0层直接pass
            return CFLAG,newNetList
        prob = np.random.choice(np.arange(0.3,1.05,0.05))
        newLayer = ['drop',prob]
        netDepartTmp[layerIdx] = newLayer  #替换
        CFLAG = judgeConv(netDepartTmp)   #判断
        if CFLAG:
            newNetList = mergeLayers(netDepartTmp)
            return CFLAG,newNetList   #返回网络正确与否的flag,新的网络(格式与popmember一样)

    elif layerType == 'fc':
        if layerIdx == 0:   #第0层直接pass
            return CFLAG,newNetList
        for i in netDepartTmp[layerIdx+1:-1]:    #fc后面不能有conv或者pool,且不考虑当前idx(将被替换)
            if (i[0] == 'pool') or (i[0] == 'conv'):
                return CFLAG,newNetList
        fcUnits = np.random.choice(np.arange(10,101))
        actType = np.random.choice(['linear','sigmoid','softmax','relu'])
        newLayer = ['fc',fcUnits,actType]
        netDepartTmp[layerIdx] = newLayer
        CFLAG = judgeConv(netDepartTmp)   #判断
        if CFLAG:
            newNetList = mergeLayers(netDepartTmp)
            return CFLAG,newNetList   #返回网络正确与否的flag,新的网络(格式与popmember一样)

def mutateAddLayer(netList,layerIdx,layerType):
    '''
    增加操作,需要检查增加后的网络合理性
    注意. 增加层操作，layeridx可以是在最后一层输出层的位置,如hiddenlayernum=5(0,1,2,3,4) 
    lyeridx可以为5 即最后一层输出层 后移 一位
    argvs:
        netList:popmember[1]
        layerIdx:需要操作的index
        layerType:将要替换的层的类型
    '''
    assert layerType in ['conv','pool','drop','fc']
    netDepartTmp = departLayers(netList)
    # netDepart = netDepartTmp[:-1] #去除最后一层fc
    CFLAG = False
    newNetList = None
    if layerType == 'conv':
        for i in netDepartTmp[:layerIdx]:    
            if i[0] == 'fc':     #卷积不能在fc后面
                return CFLAG,newNetList
        FilterNum = np.random.choice(np.arange(10,101))  
        conkSize = np.random.choice(np.arange(1,7))
        actType = np.random.choice(['linear','leaky relu','prelu','relu'])
        newLayer = ['conv',FilterNum,conkSize,actType]
        netDepartTmp.insert(layerIdx,newLayer)    #insert插入,len+1(后移)
        CFLAG = judgeConv(netDepartTmp)   #判断
        if CFLAG:
            newNetList = mergeLayers(netDepartTmp)
            return CFLAG,newNetList   #返回网络正确与否的flag,新的网络(格式与popmember一样)

    elif layerType == 'pool':  #池化层
        if layerIdx == 0:   #第0层直接pass
            return CFLAG,newNetList
        for i in netDepartTmp[:layerIdx]:
            if i[0] == 'fc':    #池化不能在fc后面
                return CFLAG,newNetList
        poolkSize = np.random.choice(np.arange(1,7))
        newLayer = ['pool',poolkSize]
        netDepartTmp.insert(layerIdx,newLayer)
        CFLAG = judgeConv(netDepartTmp)   #判断
        if CFLAG:
            newNetList = mergeLayers(netDepartTmp)
            return CFLAG,newNetList   #返回网络正确与否的flag,新的网络(格式与popmember一样)

    elif layerType == 'drop':
        if layerIdx == 0:   #第0层直接pass
            return CFLAG,newNetList
        prob = np.random.choice(np.arange(0.3,1.05,0.05))
        newLayer = ['drop',prob]
        netDepartTmp.insert(layerIdx,newLayer)
        CFLAG = judgeConv(netDepartTmp)   #判断
        if CFLAG:
            newNetList = mergeLayers(netDepartTmp)
            return CFLAG,newNetList   #返回网络正确与否的flag,新的网络(格式与popmember一样)

    elif layerType == 'fc':
        if layerIdx == 0:   #第0层直接pass
            return CFLAG,newNetList
        for i in netDepartTmp[layerIdx:-1]:    #fc后面不能有conv或者pool,且要考虑当前idx
            if (i[0] == 'pool') or (i[0] == 'conv'):
                return CFLAG,newNetList
        fcUnits = np.random.choice(np.arange(10,101))
        actType = np.random.choice(['linear','sigmoid','softmax','relu'])
        newLayer = ['fc',fcUnits,actType]
        netDepartTmp.insert(layerIdx,newLayer)
        CFLAG = judgeConv(netDepartTmp)   #判断
        if CFLAG:
            newNetList = mergeLayers(netDepartTmp)
            return CFLAG,newNetList   #返回网络正确与否的flag,新的网络(格式与popmember一样)

def mutateDelLayer(netList,layerIdx,layerType):
    '''
    删除操作,需要检查删除后的网络合理性
    argvs:
        netList:popmember[1]
        layerIdx:需要操作的index
        layerType:并没有什么用，但保持一致，代码比较好写
    '''
    assert layerType in ['conv','pool','drop','fc']
    netDepartTmp = departLayers(netList)
    # netDepart = netDepartTmp[:-1] #去除最后一层fc
    CFLAG = False
    newNetList = None
    netDepartTmp.pop(layerIdx)
    CFLAG = judgeConv(netDepartTmp)   #判断
    if CFLAG:
        newNetList = mergeLayers(netDepartTmp)
        return CFLAG,newNetList   #返回网络正确与否的flag,新的网络(格式与popmember一样)


def judgeConv(departLayers):
    '''
    判断departLayer中每个卷积大层是否是合理的
    主要用于变异时，网络层数合理性的判断
    return:
        True or False
    '''
    allConv = []
    convBig = [] #卷积大层包括:卷积层\池化层\dropout
    for layer in departLayers:
        if layer[0] == 'conv':
            if len(convBig) == 0:
                convBig.append(layer[0])
            else:
                allConv.append(convBig)  #将之前的bigconv加入大列表中
                convBig = []
                convBig.append(layer[0])
        elif (layer[0] == 'fc') or (layer[0] == 'lastfc'):
            allConv.append(convBig)
            break   #fc或者lastfc后面的不管
        else: #加进pool和drop
            convBig.append(layer[0])
    print(allConv)

    for everyConv in allConv:
        poolnum = 0
        if 'conv' not in everyConv: #没有卷积
            return False
        for l in everyConv:
            if l == 'pool':
                poolnum += 1
            if poolnum > 1 :  #池化层数大于一
                return False
        if (poolnum == 1) and ('drop' in everyConv):  #pool和drop同时存在
            poolIdx = everyConv.index('pool')
            if 'drop' in everyConv[:poolIdx]:  #drop在pool前面
                return False             
    return True    #没问题就返回True

if __name__ == '__main__':
    test = Genetic()
    pop = test.createPopulation(100)
    pass