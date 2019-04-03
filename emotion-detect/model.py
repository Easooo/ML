#author:Easooo
#data:2019-04-01

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorboard
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import torchvision as tv
import torch.optim as optim
from fer import Image,np
import os

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out) #之所以使用F.relu是因为nn.ReLU必须添加到网络结构容器中才可以使用
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidualBlock, channels=64,  num_blocks=2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, channels=128, num_blocks=2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, channels=256, num_blocks=2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, channels=512, num_blocks=2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(ResidualBlock)

def train_model(epoch,trainLoader,testLoader,savePath=None):
    device = torch.device("cpu")
    net = ResNet18()
    LR = 0.001
    optimizer = optim.SGD(net.parameters(),lr=LR,momentum=0.9)
    loss_func = nn.CrossEntropyLoss()
    for ep in range(epoch):        
        sumLoss = 0
        acc = 0
        net.train()
        for i,trainData in enumerate(trainLoader):
            inputData,targets = trainData
            inputData,targets = inputData.to(device),targets.to(device)
            optimizer.zero_grad()
            outputTrain = net(inputData)
            loss = loss_func(outputTrain,targets)
            loss.backward()
            optimizer.step()
            sumLoss += loss

            #每20个batch算一次平均loss:
            if (i % 20 == 0):
                print("Batch:%f,loss:%f"%(i,sumLoss/20))
                sumLoss = 0
        
        totalImg = 0
        accImg = 0
        with torch.no_grad():
            for testData in testLoader:
                net.eval()
                inputDataTest,labels = testData
                inputDataTest,labels = inputDataTest.to(device),labels.to(device)
                outputTest = net(inputDataTest)
                _ , predicted = torch.max(outputTest,-1)
                totalImg +=  labels.size(0)
                accImg += (predicted == outputTest).sum()        
            print("acc: %f %%" %((accImg/totalImg)*100))
            totalImg = 0
            accImg = 0

        if savePath is not None:
            torch.save(net.state_dict(), savePath)
            print('Restore model sucsses!!')


def valid_model(savePath,filePath):
    net = ResNet18()
    net.load_state_dict(torch.load(savePath))
    classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
    fileDir = os.listdir(filePath)
    assert fileDir is ['0', '1', '2', '3', '4', '5', '6']
    while True:
        emotionDir = np.random.choice(fileDir)
        print("emotion classes:",classes[int(emotionDir)])
        emotionFiles = os.listdir(emotionDir)
        imgFiles = np.random.choice(emotionFiles)
        img = Image.open(filePath+'/'+emotionDir+'/'+imgFiles)
        imgArray = np.array(img)
        imgArray = np.multiply(imgArray,1/255) 
        imgArray = imgArray[np.newaxis,np.newaxis,:]
        imgTensor = torch.from_numpy(imgArray)
        del imgArray
        with torch.no_grad():
            net.eval()
            output = net(imgTensor)
            _ , predicted = torch.max(output,-1)
            print("predict:",classes[predicted[0]])
            img.show()
        key = input()
        print("press 'exit' to exit or any key to continute")
        if key == 'exit':
            break
