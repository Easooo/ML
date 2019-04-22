#author:Easo
#date:2019年4月18日

from networks import *
from dataSet import dataLoader
from Ga import Genetic
import os
import pandas as pd 


def trainStep(popMember,popIndex,inputSize,inputChannel,epoch):
    '''
    注意:softmax和其它激活函数 使用的损失函数不一样！需要额外判断
    '''

    netList = popMember[1]  #网络列表
    lossFunc = nn.NLLLoss() if netList[-1][-1] == 'softmax' else nn.CrossEntropyLoss()
    lossFunc.cuda()  
    transformMnist = tv.transforms.ToTensor()
    trainLoader,testloader =  dataLoader(transformMnist,dataType='MNIST',batchSize=64)
    autoNet = makeNet(popMember,inputSize,inputChannel)
    # device = torch.device("cpu")
    optimizer = optim.Adam(autoNet.parameters(),lr=popMember[2])   #ADAM优化器
    autoNet.cuda()

    for ep in range(epoch):
        sum_loss = 0
        autoNet.train()
        for i,train_data in enumerate(trainLoader):
            input_data,labels = train_data  #train_data 是一个元组
            # input_data,labels = input_data.to(device),labels.to(device)
            input_data,labels = input_data.cuda(),labels.cuda()
            optimizer.zero_grad()
            output = autoNet(input_data)
            loss = lossFunc(output,labels)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()

            if (i % 100 == 0) and (i != 0):
                print("loss:%f,epoch:%d,popindex:%d,last act:%s"%(sum_loss/100,ep,popIndex,netList[-1][-1]))
                sum_loss = 0
        #计算准确率
        
        with torch.no_grad():
            acc = 0
            total = 0
            autoNet.eval()  #使dropout层进入评估状态
            for test_data in testloader:
                input_test,labels_test = test_data
                # input_test,labels_test = input_test.to(device),labels_test.to(device)
                input_test,labels_test = input_test.cuda(),labels_test.cuda()
                output_test = autoNet(input_test)
                _, predicted = torch.max(output_test, 1)  #输出得分最高的类
                total += labels_test.size(0) #统计50个batch 图片的总个数
                acc += (predicted == labels_test).sum()  #统计50个batch 正确分类的个数
            acc = acc.item()/total
            print("acc:%f"%(acc))

    saveDir = './model/' + str(popIndex)
    os.mkdir(saveDir)
    totalPara = sum(p.numel() for p in autoNet.parameters())
    popMember[-1][0],popMember[-1][1] = acc,totalPara
    torch.save(autoNet.state_dict(), saveDir+'/'+'ckp-'+str(epoch)+'-'+str(acc)+'-'+str(totalPara)+'-'+'.pth')


def makeNet(popMember,inputSize,inputChannel):
    '''
    生成网络类
    '''
    net = AutoNet(popMember,inputSize,inputChannel)
    return net

if __name__ == "__main__":
    test = Genetic()
    testPop = test.createPopulation(100)

    nHiden = [i[0] for i in testPop]
    netList = [i[1] for i in testPop]
    lrList = [i[2] for i in testPop]
    accParaList = [i[3] for i in testPop]

    saveDict = {'layernums':nHiden,
                'net':netList,
                'lr':lrList,
                'acc&ParaList':accParaList
    }
    df = pd.DataFrame(saveDict)
    df.to_csv('./pop.csv')

    for index,popMember in enumerate(testPop):
        trainStep(popMember,index,28,1,4)  




