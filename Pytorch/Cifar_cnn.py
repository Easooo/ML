import torch
import torchvision as tv
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

#      n + 2*p - k
# o =  ————————————  + 1
#          s 

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,6,5,padding=2), #输入为3*32*32 输出为6*32*32c
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv2 = nn.Sequential(nn.Conv2d(6,16,5),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2,stride=2) 
        )
        self.conv3 = nn.Sequential(nn.Conv2d(16,24,3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.fc1 = nn.Sequential(nn.Linear(24*2*2,120),#16*6*6
                                 nn.ReLU()
        )
        self.fc2 = nn.Sequential(nn.Linear(120,84),
                                 nn.ReLU()
        )
        self.fc3 = nn.Linear(84,10)

    def forward(self,input_data):
        conv1_tmp = self.conv1(input_data)
        conv2_tmp = self.conv2(conv1_tmp)
        conv3_tmp = self.conv3(conv2_tmp)
        # conv2tmp_transform = conv2_tmp.view(-1,16*6*6)  
        conv3_tmp_t = conv3_tmp.view(-1,24*2*2)
        fc1_tmp = self.fc1(conv3_tmp_t)
        fc2_tmp = self.fc2(fc1_tmp)
        res = self.fc3(fc2_tmp)
        return res

def CifarDataLoad():
    '''
    RandomCrop是裁剪，并补零
    RandomHorizontalFlip是随机翻转
    Normalize是归一化操作，后面的数字是RGB三个通道的mean和stdev（根据数据集抽样计算而得）
    '''
    transform_for_train = transforms.Compose([transforms.RandomCrop(32,padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4914,0.4822,0.4465),(0.2023, 0.1994, 0.2010)),

    ])
    transform_for_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914,0.4822,0.4465),(0.2023, 0.1994, 0.2010))                     

    ])
    train_data = tv.datasets.CIFAR10(root='./data/',train=True,transform=transform_for_train,download=True)
    test_data = tv.datasets.CIFAR10(root='./data/',train=False,transform=transform_for_test,download=True)
    trainloader = torch.utils.data.DataLoader(dataset=train_data,batch_size=32,shuffle=True)
    testloader = torch.utils.data.DataLoader(dataset=test_data,batch_size=32,shuffle=False)
    return trainloader,testloader
  
def train_step(net,trainloader,testloader,epoch):
    class_label = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
    loss_func = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    acc = 0
    # for ep in range(epoch):
    while acc < 0.1:
        acc = 0
        sum_loss = 0
        for i,train_data_loader in enumerate(trainloader):
            train_data,labels = train_data_loader
            train_data,labels = train_data.to(device),labels.to(device)
            optimizer.zero_grad()
            output_data = net(train_data)
            loss = loss_func(output_data,labels)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()

            if i % 100 == 0:
                print('loss:',sum_loss/100)
                sum_loss = 0
    
        #每一个epoch 算一次正确率
        total = 0
        for test_data in testloader:
            input_test,labels_test = test_data
            input_test,labels_test = input_test.to(device),labels_test.to(device)
            output_test = net(input_test)
            _, predicted = torch.max(output_test, 1)  #输出得分最高的类
            total += labels_test.size(0) #统计32个batch 图片的总个数
            acc += (predicted == labels_test).sum()  #统计50个batch 正确分类的个数
        acc = acc.item()/total
        print("acc:%f"%(acc))

def showimg(imgdata):
    img = imgdata[0]
    img = img.numpy()
    img = np.transpose(img, (1,2,0))
    plt.imshow(img)
    plt.show()

def detected(net):
    class_label = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    transform_for_detected = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914,0.4822,0.4465),(0.2023, 0.1994, 0.2010))                     

    ])
    detect_data = tv.datasets.CIFAR10(root='./data/',train=False,transform=transform_for_detected,download=False)
    detectloader = torch.utils.data.DataLoader(dataset=detect_data,batch_size=1,shuffle=True)
    for detect,labels in detectloader:
        output_detect = net(detect)
        _, predicted = torch.max(output_detect, 1)
        if (predicted == labels) :
            print("正确!该图片为 %s 预测结果为 %s "%(class_label[labels[0]],class_label[predicted[0]]))
        else:
            print("错误!该图片为 %s 预测结果为 %s "%(class_label[labels[0]],class_label[predicted[0]]))
        showimg(detect)
        keyboard = input()
        if keyboard == 'exit':
            break
        

def main():
    device = torch.device("cpu")
    net = CNNNet()
    net = CNNNet().to(device)
    trainloader,testloader= CifarDataLoad()
    train_step(net,trainloader,testloader,5)
    detected(net)



if __name__ == '__main__':
    # trainloader,testloader = CifarDataLoad()
    # to_pil_image = transforms.ToPILImage()
    # for train_data,labels in trainloader:
    #     print(train_data)
    #     print(labels)

    #     break
    # img_tmp = train_data[0]
    # img_tmp = img_tmp.numpy()
    # img_tmp = np.transpose(img_tmp, (1,2,0))
    # plt.imshow(img_tmp)
    # plt.show()
    # # img_tmp = train_data[0]
    # # img = to_pil_image(img_tmp)
    # # img.show()
    main()

