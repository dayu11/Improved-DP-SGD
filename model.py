import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class ConvNetTest(nn.Module):
    def __init__(self):
        super(ConvNetTest, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, kernel_size=5)
        self.conv2 = nn.Conv2d(40, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 10, kernel_size=3)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.linear1 = nn.Linear(54, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 7)

    def forward(self, x):
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        x = self.linear3(x)
        return F.log_softmax(x, dim=1)

class LinearMnist(nn.Module):
    def __init__(self):
        super(LinearMnist, self).__init__()
        self.linear1 = nn.Linear(784, 50)
        self.linear3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.tanh(self.linear1(x))
        #x = F.tanh(self.linear3(x))
        x = self.linear3(x)
        return F.log_softmax(x, dim=1)

class Logistic(nn.Module):
    def __init__(self, dataset):
        super(Logistic, self).__init__()
        if(dataset=='mnist'):
            in_dim=784
            out_dim=10
        else:
            in_dim=54
            out_dim=7
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=1)
"""
class LogisticM(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        for i in range(10):
            self.
""" 
class ConvNet64(nn.Module):
    def __init__(self):
        super(ConvNet64, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1600, 384)
        self.fc2 = nn.Linear(384, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)