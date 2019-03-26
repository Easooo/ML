import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorboard
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import argparse
import torchvision as tv
import torch.optim as optim
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
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
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


# dummy_input = torch.rand(1, 3, 28, 28) #假设输入13张1*28*28的图片
# model = ResNet18()
# with SummaryWriter(comment='ResNet18') as w:
#     w.add_graph(model, dummy_input)

def trian_step(net,trainloader,testloader,epoch):
    # 超参数设置
    EPOCH = 135   #遍历数据集次数
    pre_epoch = 0  # 定义已经遍历数据集的次数
    BATCH_SIZE = 128      #批处理尺寸(batch_size)
    LR = 0.01        #学习率
    device = torch.device("cpu")
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
    loss_func = nn.CrossEntropyLoss()
    for ep in range(epoch):
        acc_img = 0
        sum_loss = 0
        net.train()
        for i,traindata in enumerate(trainloader):
            traindata_input,labels = traindata
            traindata_input,labels = traindata_input.to(device),labels.to(device)
            optimizer.zero_grad()
            output_train = net(traindata_input)
            loss = loss_func(output_train,labels)
            loss.backward()
            optimizer.step()
            sum_loss += loss
            
            if (i % 10) == 0 :
                print("loss:",sum_loss/10)
                sum_loss = 0
        
        total_img = 0
        with torch.no_grad():
            for testdata in testloader:
                net.eval()
                testdata_input,labels_test = testdata
                testdata_input,labels_test = testdata_input.to(device),labels_test.to(device)
                output_test = net(testdata_input)
                _ , predicted = torch.max(output_test,-1)
                total_img += labels_test.size(0)
                acc_img += (predicted == labels_test).sum()
            
            print("acc:%f"%(acc_img/total_img))

def main():
    trainloader,testloader = CifarDataLoad()
    net = ResNet18()
    trian_step(net,trainloader,testloader,epoch=1)

if __name__ == '__main__':
    main()






# # Cifar-10的标签


# # 模型定义-ResNet
# net = ResNet18().to(device)

# # 定义损失函数和优化方式
# criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
# optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

# # 训练
# if __name__ == "__main__":
#     best_acc = 85  #2 初始化best test accuracy
#     print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
#     for epoch in range(pre_epoch, EPOCH):
#         print('\nEpoch: %d' % (epoch + 1))
#         net.train()
#         sum_loss = 0.0
#         correct = 0.0
#         total = 0.0
#         for i, data in enumerate(trainloader, 0):
#             # 准备数据
#             length = len(trainloader)
#             inputs, labels = data
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()

#             # forward + backward
#             outputs = net(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             # 每训练1个batch打印一次loss和准确率
#             sum_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += predicted.eq(labels.data).cpu().sum()
#             print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
#                     % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))


#         # 每训练完一个epoch测试一下准确率
#         print("Waiting Test!")
#         with torch.no_grad():
#             correct = 0
#             total = 0
#             for data in testloader:
#                 net.eval()
#                 images, labels = data
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = net(images)
#                 # 取得分最高的那个类 (outputs.data的索引号)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum()
#             print('测试分类准确率为：%.3f%%' % (100 * correct / total))
#             acc = 100. * correct / total

#             if acc > best_acc:
#                 f3 = open("best_acc.txt", "w")
#                 f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
#                 f3.close()
#                 best_acc = acc

#     print("Training Finished, TotalEPOCH=%d" % EPOCH)
