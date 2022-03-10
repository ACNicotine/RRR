from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "data/lenet_mnist_model.pth"
use_cuda=True

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

# 定义LeNet模型
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

#声明 MNIST 测试数据集何数据加载
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)

# 定义我们正在使用的设备
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# 初始化网络
model = Net().to(device)

# 加载已经预训练的模型
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# 在评估模式下设置模型。在这种情况下，这适用于Dropout图层
model.eval()

def fgsm_attack(image, epsilon, data_grad):
    # 收集数据梯度的元素符号
    sign_data_grad = data_grad.sign()
    # 通过调整输入图像的每个像素来创建扰动图像
    perturbed_image = image + epsilon*sign_data_grad
    # 添加剪切以维持[0,1]范围
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回被扰动的图像
    return perturbed_image

def test( model, device, test_loader, epsilon ):

    # 精度计数器
    # correct = 0
    # adv_examples = []
    final_res= []


    # 循环遍历测试集中的所有示例
    for data, target in test_loader:

        # 把数据和标签发送到设备
        data, target = data.to(device), target.to(device)

        # 设置张量的requires_grad属性，这对于攻击很关键
        data.requires_grad = True

        # 通过模型前向传递数据
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # 如果初始预测是错误的，不打断攻击，继续
        if init_pred.item() != target.item():
            final_res.append((data,target))

            continue

        # 计算损失
        loss = F.nll_loss(output, target)

        # 将所有现有的渐变归零
        model.zero_grad()

        # 计算后向传递模型的梯度
        loss.backward()

        # 收集datagrad
        data_grad = data.grad.data

        # 唤醒FGSM进行攻击
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        final_res.append((perturbed_data,target))
   
        # 重新分类受扰乱的图像
        # output = model(perturbed_data)

        # 检查是否成功
        # final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        # if final_pred.item() == target.item():
        #     correct += 1
        #     # 保存0 epsilon示例的特例
        #     if (epsilon == 0) and (len(adv_examples) < 5):
        #         adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
        #         adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        # else:
        #     # 稍后保存一些用于可视化的示例
        #     if len(adv_examples) < 5:
        #         adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
        #         adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # 计算这个epsilon的最终准确度
    # final_acc = correct/float(len(test_loader))
    # print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # 返回准确性和对抗性示例
    print("I have finished the mission!")
    return final_res


def normal_train(model, optimizer, loader, summary_epochs):
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
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on given dataset: {} %'.format(100 * correct / total))


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
# 对每个epsilon运行测试

combine = test(model, device, test_loader, 0.2)


Model1 = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

normal_train(Model1,optimizer,test_loader,10)
verify(Model1,test_loader)    
normal_train(Model1,optimizer,combine,10)
verify(Model1,test_loader)