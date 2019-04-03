#author:Easooo
#data:2019-04-03

from model import ResNet18,train_model,valid_model
from fer import fer2013,transforms,torch


def mainTrain():
    csvPath = './data/fer2013.csv'
    transformTrain = transforms.Compose([transforms.RandomResizedCrop(42),
                                         transforms.RandomHorizontalFlip(),                                      
                                         transforms.ToTensor(),                                       
    ])
    transformTest = transforms.Compose([transforms.CenterCrop(42),                               
                                        transforms.ToTensor(),         
    ])
    trainLoader = fer2013(csvPath,batchsize=32,datatype='train',transform=transformTrain,shuffleFlag=True)
    testLoader = fer2013(csvPath,batchsize=32,datatype='test',transform=transformTest,shuffleFlag=False)
    train_model(5,trainLoader,testLoader,savePath='./model/')

if __name__ == '__main__':
    mainTrain()
