import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import onnx
import time
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained_model = "model/my_model1.pth"
class Block(nn.Module):
    def __init__(self, inchannel, outchannel, res=True, stride=1):
        super(Block, self).__init__()
        self.res = res     # 是否带残差连接
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, bias=False),
                nn.BatchNorm2d(outchannel),
            )
        else:
            self.shortcut = nn.Sequential()

        self.relu = nn.Sequential(
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.left(x)
        if self.res:
            out += self.shortcut(x)
        out = self.relu(out)
        return out


class myModel(nn.Module):
    def __init__(self, cfg=[64, 'M', 128, 'M', 256, 'M', 512, 'M'], res=True):
        super(myModel, self).__init__()
        self.res = res       # 是否带残差连接
        self.cfg = cfg       # 配置列表
        self.inchannel = 3   # 初始输入通道数
        self.futures = self.make_layer()
        # 构建卷积层之后的全连接层以及分类器：
        self.classifier = nn.Sequential(nn.Dropout(0.4),           # 两层fc效果还差一些
                                        nn.Linear(4 * 512, 10), )   # fc，最终Cifar10输出是10类

    def make_layer(self):
        layers = []
        for v in self.cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(Block(self.inchannel, v, self.res))
                self.inchannel = v    # 输入通道数改为上一层的输出通道数
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.futures(x)
        # view(out.size(0), -1): change tensor size from (N ,H , W) to (N, H*W)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

norm_mean = [0.485, 0.456, 0.406]  # 均值
norm_std = [0.229, 0.224, 0.225]  # 方差
transform_train = transforms.Compose([transforms.ToTensor(),  # 将PILImage转换为张量
                                      # 将[0,1]归一化到[-1,1]
                                      transforms.Normalize(norm_mean, norm_std),
                                      transforms.RandomHorizontalFlip(),  # 随机水平镜像
                                      transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  # 随机遮挡
                                      transforms.RandomCrop(32, padding=4)  # 随机中心裁剪
                                      ])

transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(norm_mean, norm_std)])

batch_size = 256
num_epochs = 200   # 训练轮数
LR = 0.001          # 初始学习率

# 选择数据集:
trainset = datasets.CIFAR10(root='Datasets', train=True, download=True, transform=transform_train)
testset = datasets.CIFAR10(root='Datasets', train=False, download=True, transform=transform_test)
# 加载数据:
train_data = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
valid_data = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

train_data_size = len(trainset)
valid_data_size = len(testset)



def ewc_train(model, optimizer, previous_loader,  summary_epochs, lambda_ewc,eps):
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
        for step, (imgs, labels) in enumerate(previous_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            inputs_fgm = fast_gradient_method(model, imgs, 6/255, np.inf)
            outputs = model(inputs_fgm)
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

        torch.save(model.state_dict(), './model/my_model1.pth')
               
    return losses

model = myModel(res=True)
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-3)
epsilons = [7/255,8/255,9/255,10/255,11/255,12/255,13/255,14/255,15/255,16/255]
for eps in epsilons:
    ewc_train(model,optimizer,train_data,10,10,eps)
