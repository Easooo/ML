#author:Easooo
#data:2019-03-28

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import torchvision as tv


def get_data(originCsvData,dataType='trian'):
    '''
    get ndarray from csv file
    return:
        a tuple contain imgdata and labels
    
    '''
    #trainData,testData,vaildData
    assert dataType in ('train','test','valid')
    dataUsage = {'train':'Training','test':'PublicTest','valid':'PrivateTest'}
    dataTmp = originCsvData[originCsvData['Usage'] == dataUsage[dataType]].reset_index(drop=True)
    labelsTmp = pd.get_dummies(dataTmp['emotion']).as_matrix()      #这里需要了解原文件的格式
    labelsTmp = torch.LongTensor(labelsTmp)
    _,labels = torch.max(labelsTmp,-1)       #将label转换为tensor
    pixelsList = dataTmp['pixels'].tolist()
    imgTmp = []
    #源文件中的像素是一整行的，这里需要将其reshape为48*48
    for everyPixels in pixelsList:
        img = [int (pixel) for pixel in everyPixels.split(' ')]
        img = np.array(img).reshape(48,48)
        imgTmp.append(img)
    # imgData = torch.FloatTensor(imgTmp)
    # imgData = imgData.unsqueeze(-1)
    imgData = np.array(imgTmp)
    #imgData = np.expand_dims(imgData,-1)
    # imgData = imgData.view(-1,1,48,48)
    return (imgData,labels)    #imgData.shape: (nums,48,48)


class FerDataset(data.Dataset):
    #自定义数据集
    def __init__(self,soureceData,transform=None):
        self.imgData,self.labelData = soureceData
        self.transform = transform
        
    def __len__(self):
        return self.imgData.shape[0]
        

    def __getitem__(self,index:int):
        img,labels = self.imgData[index],self.labelData[index]
        imgUint8 = np.uint8(img)
        # img3c = np.uint8(np.concatenate((img,img,img),axis=2))
        img = Image.fromarray(imgUint8)   #这里的img的mode为L，即为灰度图像
        if self.transform is not None:
            img = self.transform(img)   #通过torch.ToTensor()后,Image对象img将变成一个size为(1.48.48)
        return img,labels

def Fer_dataloader(dataset,batchsize,shuffleFlag=True):
    loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=batchsize,shuffle=shuffleFlag)
    return loader

def fer2013(path,batchsize,datatype,transform=None,shuffleFlag=True):
    '''
    combine funcs
    return:
        loader
    '''
    originData = pd.read_csv(path)
    dataTmp = get_data(originData,dataType=datatype)
    ferSet = FerDataset(dataTmp,transform)
    return  Fer_dataloader(ferSet,batchsize,shuffleFlag)



if __name__ == '__main__':
    path = './data/fer2013.csv'
    transformForTest = transforms.Compose([transforms.ToTensor(),
                                        #    transfroms.Normalize((),())

    ])
    classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
    loader = fer2013(path,16,'test',transformForTest,shuffleFlag=False)
    for l in loader:
        inputdata,label = l
        print(inputdata,label)

    