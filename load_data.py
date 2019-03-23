import numpy as np
from torchvision import datasets, transforms
import torch

class MNIST_data():
    @staticmethod
    def train():
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=60000, shuffle=True)
        for (data, target) in train_loader:#getting whole batch one time
            return (data, target)
    @staticmethod
    def test():
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=10000, shuffle=True)
        for (data, target) in train_loader:#getting whole batch one time
            return (data, target)

##data is from https://github.com/sunblaze-ucb/dpml-benchmark
class Covertype_data():
    @staticmethod
    def train():
        x=np.load('../data'+'/covertype_processed_x.npy')
        y=np.load('../data'+'/covertype_processed_y.npy')
        total=x.shape[0]
        num_trn=int(total*0.8)
        trn_x=torch.tensor(x[0:num_trn], dtype=torch.float32)
        trn_y=torch.zeros(num_trn, dtype=torch.long)
        for i in range(num_trn):
            for j in range(len(y[0])):
                if(y[i][j]==1):
                    trn_y[i]=j
        return (trn_x, trn_y)

    @staticmethod
    def test():
        x=np.load('../data'+'/covertype_processed_x.npy')
        y=np.load('../data'+'/covertype_processed_y.npy')
        total=x.shape[0]
        idx=int(total*0.8)
        num_test=int(0.2*total)
        test_x=torch.tensor(x[int(total*0.8):-1], dtype=torch.float32)
        test_y=torch.zeros(num_test, dtype=torch.long)
        for i in range(num_test):
            for j in range(len(y[0])):
                if(y[idx+i][j]==1):
                    test_y[i]=j
        return (test_x, test_y)

class CIFAR_data():
    @staticmethod
    def train():
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=True, download=True,
                        transform=transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
            batch_size=50000, shuffle=True)

        for (data, target) in train_loader:#getting whole batch one time
            return (data, target)
    @staticmethod
    def test():
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False, download=True,
                        transform=transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
            batch_size=10000, shuffle=True)

        for (data, target) in train_loader:#getting whole batch one time
            return (data, target)

