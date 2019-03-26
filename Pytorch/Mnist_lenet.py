import torch
import torchvision as tv
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np

'''
input 32*32
conv1 6*28*28  kernel: 5*5
pool1 6*14*14  kernel: 2*2  stride:2
conv2 16*10*10 kernel: 5*5
pool2 16*5*5   kernel: 2*2 stride:2
f1 120 kernel: 16*5*5
f2 84 kernel:120
output 10 kernel:84

'''
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,6,5,padding=2), #输入为1*28*28 输出为6*28*28
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2,stride=2)
        ) 
        self.conv2 =nn.Sequential(nn.Conv2d(6,16,5),#输入为6*14*14 输出为16*10*10
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2,stride=2) #input: 16*10*10 output:16*5*5
        )  
        self.fc1 = nn.Sequential(nn.Linear(16*5*5,120),
                                 nn.ReLU()
        )
        self.fc2 = nn.Sequential(nn.Linear(120,84),
                                 nn.ReLU()
        )
        self.fc3 = nn.Linear(84,10)

    def forward(self,input_data):
        #向前传播的过程
        conv1_tmp = self.conv1(input_data)
        conv2_tmp = self.conv2(conv1_tmp)
        conv2_tmp = conv2_tmp.view(conv2_tmp.size(0),-1)
        fc1_tmp = self.fc1(conv2_tmp)
        fc2_tmp = self.fc2(fc1_tmp)
        res = self.fc3(fc2_tmp)
        return res

def DataLoad():
    transform = tv.transforms.ToTensor()
    train_data = tv.datasets.MNIST(root='./data/',train=True,transform=transform,download=True)
    test_data = tv.datasets.MNIST(root='./data/',train=False,transform=transform,download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=16,shuffle=True)  #生成元组
    test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=16,shuffle=False)
    return train_loader,test_loader

def train_step(net,device,train_loader,tes_loader,epoch):
    optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
    loss_func = nn.CrossEntropyLoss()
    for ep in range(epoch):
        sum_loss = 0
        for i,train_data in enumerate(train_loader):
            input_data,labels = train_data  #train_data 是一个元组
            input_data,labels = input_data.to(device),labels.to(device)
            optimizer.zero_grad()
            output = net(input_data)
            loss = loss_func(output,labels)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()

            if i % 100 == 0:
                print("loss:%f,epoch:%d"%(sum_loss/100,ep))
                sum_loss = 0
        #计算准确率

        acc = 0
        total = 0
        for test_data in tes_loader:
            input_test,labels_test = test_data
            input_test,labels_test = input_test.to(device),labels_test.to(device)
            output_test = net(input_test)
            _, predicted = torch.max(output_test, 1)  #输出得分最高的类
            total += labels_test.size(0) #统计50个batch 图片的总个数
            acc += (predicted == labels_test).sum()  #统计50个batch 正确分类的个数
        print("acc:%f"%(acc.item()/total))
    
def main():
    device = torch.device("cpu")
    net = LeNet().to(device)
    train_loader,test_loader = DataLoad()
    train_step(net,device,train_loader,test_loader,epoch=1)

if __name__ == '__main__':
    # # model = LeNet()
    # # conv1 = model.conv1
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    main()


