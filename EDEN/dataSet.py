#author: Easo
#date: 2019年4月12日
"""
Creat DataLoader of Minist & Cifar-10
"""
import torch
import torchvision.transforms as transforms
import torchvision as tv

def dataLoader(transform,dataType='MNIST',trainBatchSize=16,testBatchSize=16):
    '''
    argvs:
        transform:数据转换格式
        dataType:数据集类型
        batchSize:每个Batch的大小
    return: 
        trainloader和testloader
        
    '''
    dataSetDict = {'MNIST':tv.datasets.MNIST,'CIFAR10':tv.datasets.CIFAR10}
    assert dataType in ['MNIST','CIFAR10']
    dataSetType = dataSetDict[dataType]
    trainData = dataSetType(root='./data/',train=True,transform=transform,download=True)
    testData = dataSetType(root='./data/',train=False,transform=transform,download=True)
    trainloader = torch.utils.data.DataLoader(dataset=trainData,batch_size=trainBatchSize,shuffle=True)
    testloader = torch.utils.data.DataLoader(dataset=testData,batch_size=testBatchSize,shuffle=False)
    return trainloader,testloader
