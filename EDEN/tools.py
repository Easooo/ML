#author:Easo
#date:2019年5月9日

from networks import *
from networks import torch
import train
import os
from dataSet import allTransformList
import copy
from dataSet import dataLoader

def advanceTrian(popMember,inputSize,inputChannel,epoch,transformsList,dataType,oldModelDir=None):
    '''
    注意:softmax和其它激活函数 使用的损失函数不一样！需要额外判断
    '''

    netList = popMember[1]  #网络列表
    lossFunc = nn.NLLLoss() if netList[-1][-1] == 'softmax' else nn.CrossEntropyLoss()
    lossFunc.cuda()  
    trainLoader,testloader = dataLoader(transformsList[0],transformsList[1],dataType=dataType,trainBatchSize=256,testBatchSize=64)
    autoNet = makeNet(popMember,inputSize,inputChannel)
    if oldModelDir is not None:
        autoNet.load_state_dict(torch.load(oldModelDir))
        print('load model dir:',oldModelDir)
    # device = torch.device("cpu")
    optimizer = optim.Adam(autoNet.parameters(),lr=popMember[2])   #ADAM优化器
    autoNet.cuda()

    oldLoss = False
    currentLoss = False

    returnEp = epoch
    for ep in range(epoch):
        sum_loss = 0
        lossTmp = 0
        autoNet.train()
        for i,train_data in enumerate(trainLoader):
            input_data,labels = train_data  #train_data 是一个元组
            input_data,labels = input_data.cuda(),labels.cuda()
            optimizer.zero_grad()
            output = autoNet(input_data)
            loss = lossFunc(output,labels)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            if (i % 100 == 0) and (i != 0):
                print("loss:%f,epoch:%d,popindex:%d,last act:%s"%(sum_loss/100,ep,popMember[7],netList[-1][-1]))
                lossTmp = sum_loss/100
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

            if ep == 0:
                currentLoss = lossTmp
            else:
                oldLoss = currentLoss
                currentLoss = lossTmp
                print("old loss:%f,current loss:%f"%(oldLoss,currentLoss))
                value = abs(oldLoss-currentLoss)
                if value < 1e-5:
                    returnEp = ep
                    break

    totalPara = sum(p.numel() for p in autoNet.parameters())
    popMember[3],popMember[4] = acc,totalPara
    return autoNet,returnEp

def makeNet(popMember,inputSize,inputChannel):
    '''
    生成网络类
    '''
    net = AutoNet(popMember,inputSize,inputChannel)
    return net

if __name__ == "__main__":
    dataType = input('input dataType:')

    inputSize = input('inputsize:')
    inputSize = int(inputSize)

    inputChannel = input('inputchannel:')
    inputChannel = int(inputChannel)
    tfList = allTransformList(dataType)

    popNetList = [
[[0, None, None, 55, 5, 'prelu'], [0, None, None, 58, 5, 'relu'], [0, None, 2, 82, 6, 'leaky relu'], [0, None, 5, 58, 2, 'leaky relu'], [0, None, 4, 44, 4, 'linear'], [1, None, 63, 'relu'], [2, 10, 'softmax']],
[[0, None, 5, 61, 5, 'linear'], [0, None, None, 37, 5, 'relu'], [2, 10, 'softmax']],
[[0, None, None, 22, 4, 'leaky relu'], [0, None, 3, 78, 1, 'leaky relu'], [0, None, None, 80, 4, 'relu'], [2, 10, 'softmax']]
    ]
    modelList = [
'9-0.7901-319314-1.2098968682863889.pth',
'9-0.7262-74624-1.2737865994854203.pth',
'9-12-0.7641-218362-1.235895420448613.pth'
    ]
    popMemberList = []
    popMemberList.append([9,popNetList[0],0.00112517910583453,0.7901,319314,1.20989686828638,['lr', 'del', 'add'],80])
    popMemberList.append([3,popNetList[1],0.00915443183972667,0.7262,74624,1.27378659948542,[],20])
    popMemberList.append([4,popNetList[2],0.00647553002829942,0.7484,218362,1.23589542044861,[],37])

    for idx,popMember in enumerate(popMemberList):
        with open('./advancedTarin/oldModelLog.txt','a+') as f:
            f.write(str(popMember)+'\n')
            f.write('\n')
            f.close()
        modeldir = './advancedTarin/' + modelList[idx]
        popadvancedNet,returnEp = advanceTrian(popMember,inputSize,inputChannel,300,tfList,dataType,modeldir)
        saveName = str(popMember[3])+'-'+str(popMember[4])+'-'+str(popMember[5])+'.pth'
        os.mkdir('./advancedTarin/'+str(idx))
        saveDir = './advancedTarin/'+str(idx)
        torch.save(popadvancedNet.state_dict(), saveDir+'/'+saveName)
        with open('./advancedTarin/newModelLog.txt','a+') as g:
            g.write(str(popMember)+str(returnEp)+'\n')
            g.write('\n')
            g.close()
