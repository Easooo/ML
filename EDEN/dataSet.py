#author: Easo
#date: 2019年4月12日
"""
Creat DataLoader of Minist & Cifar-10
"""
import torch
import torchvision.transforms as transforms
import torchvision as tv

def dataLoader(transformTrain,transformsTest,dataType,trainBatchSize=16,testBatchSize=16):
    '''
    argvs:
        transform:数据转换格式
        dataType:数据集类型
        batchSize:每个Batch的大小
    return: 
        trainloader和testloader
        
    '''
    dataSetDict = {'MNIST':tv.datasets.MNIST,'CIFAR10':tv.datasets.CIFAR10,'FMNIST':tv.datasets.FashionMNIST}
    assert dataType in ['MNIST','CIFAR10','FMNIST']
    dataSetType = dataSetDict[dataType]
    trainData = dataSetType(root='./data/',train=True,transform=transformTrain,download=True)
    testData = dataSetType(root='./data/',train=False,transform=transformsTest,download=True)
    trainloader = torch.utils.data.DataLoader(dataset=trainData,batch_size=trainBatchSize,shuffle=True)
    testloader = torch.utils.data.DataLoader(dataset=testData,batch_size=testBatchSize,shuffle=False)
    return trainloader,testloader

def allTransformList(dataType):
    '''
    数据格式
    '''
    assert dataType in ['MNIST','CIFAR10','FMNIST']
    tList = []
    if dataType == 'CIFAR10':
        transform_for_train = transforms.Compose([transforms.RandomCrop(32,padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914,0.4822,0.4465),(0.2023, 0.1994, 0.2010)),
        ])
        tList.append(transform_for_train)
        transform_for_test = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.4914,0.4822,0.4465),(0.2023, 0.1994, 0.2010))                     
        ])
        tList.append(transform_for_test)
        return tList
    else:
        transforMnist = tv.transforms.ToTensor()
        tList.append(transforMnist)
        tList.append(transforMnist)
        return tList

        
if __name__ == "__main__":
    t = tv.datasets.MNIST(root='./data/',train=True,transform=tv.transforms.ToTensor(),download=True)
    pass
    torch.utils.data.sampler.SequentialSampler