import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.data.sampler as sampler
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import os
import random
from copy import deepcopy
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datapath = './results'
txtpath = './results/test.txt'

datapath2 = './results0.04'
txtpath2 = './results0.04/test.txt'

pretrained_model = "data/lenet_mnist_model.pth"
epsilons = [0, 0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15]
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class MyDataset(Dataset):
    def __init__(self,txtpath):
        #创建一个list用来储存图片和标签信息
        imgs = []
        #打开第一步创建的txt文件，按行读取，将结果以元组方式保存在imgs里
        datainfo = open(txtpath,'r')
        for line in datainfo:
            line = line.strip('\n')
            words = line.split()
            imgs.append((words[0],words[1]))

        self.imgs = imgs
	#返回数据集大小
    def __len__(self):
        return len(self.imgs)
	#打开index对应图片进行预处理后return回处理后的图片和标签
    def __getitem__(self, index):
        pic,label = self.imgs[index]
        pic = Image.open(datapath+'/'+pic)
        print(datapath)
        pic = transforms.ToTensor()(pic)
        label = int(label)
        return pic,label
#实例化对象

#将数据集导入DataLoader，进行shuffle以及选取batch_size
dataa2 = MyDataset(txtpath2)

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(28*28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        return x



MNIST_transform = transforms.Compose([
    transforms.ToTensor(),
])
USPS_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])
mnist = torchvision.datasets.MNIST(
    root='/data',
    train=True,                                     
    transform = MNIST_transform,
    download=True
)
mnist2 = torchvision.datasets.MNIST(
    root='/data',
    train=False,                                     
    transform = MNIST_transform,
    download=True
)

batch_size = 100


data_loader2 = DataLoader(dataa2,batch_size=100,shuffle=True,num_workers=0)
mnist_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size, 
                                          shuffle=True)
mnist_loader2 = torch.utils.data.DataLoader(dataset=mnist2,
                                          batch_size=batch_size, 
                                          shuffle=True)


def normal_train(model, optimizer, loader, summary_epochs):
    model.train()
    model.zero_grad()
    loss_func = nn.CrossEntropyLoss()
    losses = []
    loss = 0.0
    for epoch in range(summary_epochs):
        for step, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), torch.tensor(labels).to(device)
            outputs = model(imgs)
            ce_loss = loss_func(outputs, labels)

            optimizer.zero_grad()
            ce_loss.backward()
            optimizer.step()

            loss += ce_loss.item()
            if (step + 1) % 20 == 0:
                loss = loss / 20
                print ("\r", "Epoch {}, step {}, loss: {:.3f}      ".format(epoch + 1,step+1,loss), end=" ")
                losses.append(loss)
                loss = 0.0
                
    return losses

def verify(model, loader):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            final_pred = outputs.max(1, keepdim=True)[1]
            if final_pred.item() == labels.item():
               correct += 1
            total += labels.size(0)
        print('Accuracy of the network on given dataset: {} %'.format(100 * correct / total))

# def verify(model, loader):
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for images, labels in loader:
#             images = images.reshape(-1, 28*28).to(device)
#             labels = labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#         print('Accuracy of the network on given dataset: {} %'.format(100 * correct / total))



def ewc_train(model, optimizer, previous_loader, loader,  summary_epochs, lambda_ewc):
    # 计算重要度矩阵
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}# 模型的所有参数
    
    _means = {} # 初始化要把参数限制在的参数域
    for n, p in params.items():
        _means[n] = p.clone().detach()
    
    precision_matrices = {} #重要度
    for n, p in params.items():
        precision_matrices[n] = p.clone().detach().fill_(0) #取zeros_like

    model.eval()
    for data, labels in previous_loader:
        model.zero_grad()
        data, labels = data.to(device),labels.to(device)
        output = model(data)
        ############ 核心代码 #############
        loss = F.nll_loss(F.log_softmax(output, dim=1), labels)
        # 计算labels对应的(正确分类的)对数概率，并把它作为loss func衡量参数重要度        
        loss.backward()  # 反向传播计算导数
        for n, p in model.named_parameters():                         
            precision_matrices[n].data += p.grad.data ** 2 / len(previous_loader)
        ########### 计算对数概率的导数，然后反向传播计算梯度，以梯度的平方作为重要度 ########

    model.train()
    model.zero_grad()
    loss_func = nn.CrossEntropyLoss()
    losses = []
    loss = 0.0
    for epoch in range(summary_epochs):
        for step, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            ce_loss = loss_func(outputs, labels)
            total_loss = ce_loss
            # 额外计算EWC的L2 loss
            ewc_loss = 0
            for n, p in model.named_parameters():
                _loss = precision_matrices[n] * (p - _means[n]) ** 2
                ewc_loss += _loss.sum()
            total_loss += lambda_ewc * ewc_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss += total_loss.item()
            if (step + 1) % 20 == 0:
                loss = loss / 20
                print ("\r", "Epoch {}, step {}, loss: {:.3f}      ".format(epoch + 1,step+1,loss), end=" ")
                losses.append(loss)
                loss = 0.0
                
    return losses

model1 = Net().to(device)

# 加载已经预训练的模型
model1.load_state_dict(torch.load(pretrained_model, map_location='cpu'))


model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#normal_train(model,optimizer,mnist_loader,10)
#verify(model,data_loader2)
#normal_train(model,optimizer,data_loader,10)
# ewc_train(model,optimizer,mnist_loader,data_loader,10,350)
# verify(model,data_loader2)
# verify(model,mnist_loader2)
for eps in epsilons:
    datapath = './results{}'.format(eps)
    txtpath = './results{}/test.txt'.format(eps)
    dataa = MyDataset(txtpath)
    data_loader = DataLoader(dataa,batch_size=100,shuffle=True,num_workers=0)
    ewc_train(model,optimizer,mnist_loader,data_loader,10,350)
    verify(model,mnist_loader2)





