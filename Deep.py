import pandas as pd
import numpy as np
import matplotlib
from matplotlib.animation import FuncAnimation
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import torchvision
import torch
import torch.nn as nn
import torch.utils.data as Data
from d2l import torch as d2l
from torchsummary import summary
import random

import matplotlib
import matplotlib.pyplot as plt

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(41, 36),
                    nn.Sigmoid(),          # activate
                    nn.Linear(36, 24),
                    nn.Sigmoid(),
                    nn.Linear(24, 12),
                    nn.Sigmoid(),
                    nn.Linear(12, 6),
                    nn.Sigmoid(),
                    nn.Linear(6, 2))

batch_size, lr, num_epochs = 500, 0.25, 10

net = net.double()
net = net.cuda()
# read csv
X_train = pd.read_csv("train.csv",
                      header=None,index_col=False,
                      names=["0","1","2","3","4","5","6","7","8","9","10",
                     "11","12","13","14","15","16","17","18","19","20",
                     "21","22","23","24","25","26","27","28","29","30",
                     "31","32","33","34","35","36","37","38","39","40",
                     "41"])
X_test = pd.read_csv("test.csv",
                     header=None,index_col=False,
                     names=["0","1","2","3","4","5","6","7","8","9","10",
                     "11","12","13","14","15","16","17","18","19","20",
                     "21","22","23","24","25","26","27","28","29","30",
                     "31","32","33","34","35","36","37","38","39","40",
                     "41"])
# relabel
X_train = X_train.dropna(how = 'any').reset_index(drop = True)
gle = LabelEncoder()
X_train[["e1","e2","e3"
         ]] = X_train[[
             "1","2","3"
         ]].apply(gle.fit_transform) 
X_test[["e1","e2","e3"
         ]] = X_test[[
             "1","2","3"
         ]].apply(gle.fit_transform)
# relabel normal and attack
y_train=[]
train_color=[]
for i in range(X_train["41"].shape[0]):
    if X_train.at[i,"41"] == 'normal.':
        y_train.append(0)
    else:
        y_train.append(1)
y_train_write=pd.DataFrame(y_train)
X_train["e41"]=y_train_write

def train_mlp(net, train_iter, test_iter, num_epochs, lr, device):
    """GPU"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, std=0.01)
            # nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr) # Optimizer
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr) # Optimizer
    loss = nn.CrossEntropyLoss() # loss
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train error', 'test error'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        #train_loss.sum, train_acc.sum, num_samples
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            #X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, 1-train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, 1-test_acc))
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
            f'test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
            f'on {str(device)}')
        print(f'epoch{epoch}finished')
    

y_test=[]
y_color=[]
for i in range(X_test["41"].shape[0]):
    if X_test.at[i,"41"] == 'normal.':
        y_test.append(0)
    else:
        y_test.append(1)
y_test_write=pd.DataFrame(y_test)

X_test["e41"]=y_test_write
x_train=np.array(X_train[["0","e1","e2","e3","4","5","6","7","8","9","10",
                     "11","12","13","14","15","16","17","18","19","20",
                     "21","22","23","24","25","26","27","28","29","30",
                     "31","32","33","34","35","36","37","38","39","40"
         ]])
x_test=np.array(X_test[["0","e1","e2","e3","4","5","6","7","8","9","10",
                     "11","12","13","14","15","16","17","18","19","20",
                     "21","22","23","24","25","26","27","28","29","30",
                     "31","32","33","34","35","36","37","38","39","40"
         ]])
x_train_tensor= torch.as_tensor(x_train,device=d2l.try_gpu())
x_test_tensor= torch.as_tensor(x_test,device=d2l.try_gpu())
y_train_tensor=torch.as_tensor(y_train_write.values,device=d2l.try_gpu()).squeeze()
y_test_tensor=torch.as_tensor(y_test_write.values,device=d2l.try_gpu()).squeeze()
train_iter = Data.DataLoader(
        dataset=Data.TensorDataset(x_train_tensor,y_train_tensor),  # Data.TensorDataset()
        batch_size=batch_size,  # batch size
        shuffle=True,  # 
        num_workers=0,  # multiprocess
    )
test_iter = Data.DataLoader(
        dataset=Data.TensorDataset(x_test_tensor, y_test_tensor),  # Data.TensorDataset()
        batch_size=batch_size,  # batch size
        shuffle=False,  # 
        num_workers=0,  # multiprocess
    )
print("------------------------------train_iter------------------------------")
print(x_train_tensor)
print(y_train_tensor)
print("------------------------------test_iter-------------------------------")
print(x_test_tensor)
print(y_test_tensor)
print("-----------------------net.parameters.device---------------------------")
print(list(net.parameters())[0].device)

devices = d2l.try_all_gpus()
train_mlp(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu()) 
d2l.plt.savefig('normal_sigmoid_SGD.png',bbox_inches = 'tight',dpi=300)
print("-------------------------------net--------------------------------------")

print(net)
print("Activation function = Sigmoid, parameter initialization method = normal_, optimizer = SGD, loss function = CELoss")
print("--------------------------net.parameters-------------------------------")

for name, param in net.named_parameters():
    print(name)
    print(param.data)
    print("requires_grad:", param.requires_grad)
    print("-------------------------------------------------------------------------")
print("------------------------------summary-----------------------------------")
print(summary(net))
