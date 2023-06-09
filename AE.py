import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.utils.data as Data
from loguru import logger
from d2l import torch as d2l



# AutoEncoder
class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_layer_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        # input size = output size
        self.input_size = input_size
        self.output_size = input_size

        self.encode_linear = nn.Linear(self.input_size, hidden_layer_size)
        self.decode_linear = nn.Linear(hidden_layer_size, self.output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_x):
        # encode
        encode_linear = self.encode_linear(input_x)
        encode_out = self.relu(encode_linear)
        # decode
        decode_linear = self.decode_linear(encode_out)  # =self.linear(lstm_out[:, -1, :])
        return decode_linear
def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            # nn.init.normal_(m.weight, std=0.01)
            nn.init.xavier_uniform_(m.weight)

# train
def train_auto_encoder(normal_data: np.ndarray):
    """train Auto Encoder"""
    train_tensor = torch.tensor(normal_data).float()
    batch_size = 2000

    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(train_tensor),  # Data.TensorDataset()
        batch_size=batch_size,  # batch size
        shuffle=True,  # 
        num_workers=0,  # multiprocess
    )
    # loss， optimizor，epochs
    model = AutoEncoder(train_tensor.shape[1])  # model
    
    model.apply(init_weights)
    loss_function = nn.MSELoss()  #loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5)  # optimizor
    epochs = 15
    # start training
    model.train()
    loss_list = []
    for i in range(epochs):
        epoch_loss_list = []
        for seq in train_loader:
            seq = seq[0]
            optimizer.zero_grad()
            y_pred = model(seq).squeeze()  # output
            single_loss = loss_function(y_pred, seq)
            single_loss.backward()
            optimizer.step()
            epoch_loss_list.append(single_loss.detach().numpy())
        logger.debug("Train Step:{} loss: {}", i, np.mean(epoch_loss_list))
        loss_list.append(np.mean(epoch_loss_list))
    return model, np.min(loss_list)

# pca = PCA(n_components=2)

# read csv
X_train = pd.read_csv("kddcup99_train.csv",
                      header=None,index_col=False,
                      names=["0","1","2","3","4","5","6","7","8","9","10",
                     "11","12","13","14","15","16","17","18","19","20",
                     "21","22","23","24","25","26","27","28","29","30",
                     "31","32","33","34","35","36","37","38","39","40",
                     "41"])
X_test = pd.read_csv("kddcup99_test.csv",
                     header=None,index_col=False,
                     names=["0","1","2","3","4","5","6","7","8","9","10",
                     "11","12","13","14","15","16","17","18","19","20",
                     "21","22","23","24","25","26","27","28","29","30",
                     "31","32","33","34","35","36","37","38","39","40",
                     "41"])
# relabel
X_train = X_train.dropna(how = 'any').reset_index(drop = True)
gle = LabelEncoder()
X_train[["e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","e10",
         "e11","e12","e13","e14","e15","e16","e17","e18","e19","e20",
         "e21","e22","e23","e24","e25","e26","e27","e28","e29","e30",
         "e31","e32","e33","e34","e35","e36","e37","e38","e39","e40"
         ]] = X_train[[
                     "0","1","2","3","4","5","6","7","8","9","10",
                     "11","12","13","14","15","16","17","18","19","20",
                     "21","22","23","24","25","26","27","28","29","30",
                     "31","32","33","34","35","36","37","38","39","40"
         ]].apply(gle.fit_transform) 
X_test[["e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","e10",
         "e11","e12","e13","e14","e15","e16","e17","e18","e19","e20",
         "e21","e22","e23","e24","e25","e26","e27","e28","e29","e30",
         "e31","e32","e33","e34","e35","e36","e37","e38","e39","e40"
         ]] = X_test[[
                     "0","1","2","3","4","5","6","7","8","9","10",
                     "11","12","13","14","15","16","17","18","19","20",
                     "21","22","23","24","25","26","27","28","29","30",
                     "31","32","33","34","35","36","37","38","39","40"
         ]].apply(gle.fit_transform)
# relabel normal and attack
y_train=[]
train_color=[]
for i in range(X_train["41"].shape[0]):
    if X_train.at[i,"41"] == 'normal.':
        y_train.append(0)
        train_color.append('blue') # blue -> normal
    else:
        y_train.append(1)
        train_color.append('orange') # orange -> attack
y_train_write=pd.DataFrame(y_train)
X_train["e41"]=y_train_write

y_test=[]
y_color=[]
for i in range(X_test["41"].shape[0]):
    if X_test.at[i,"41"] == 'normal.':
        y_test.append(0)
        y_color.append('blue') # blue -> normal
    else:
        y_test.append(1)
        y_color.append('orange') # orange -> attack
y_test_write=pd.DataFrame(y_test)

X_test["e41"]=y_test_write
x_train=np.array(X_train[["e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","e10",
         "e11","e12","e13","e14","e15","e16","e17","e18","e19","e20",
         "e21","e22","e23","e24","e25","e26","e27","e28","e29","e30",
         "e31","e32","e33","e34","e35","e36","e37","e38","e39","e40"
         ]])



x_test=np.array(X_test[["e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","e10",
         "e11","e12","e13","e14","e15","e16","e17","e18","e19","e20",
         "e21","e22","e23","e24","e25","e26","e27","e28","e29","e30",
         "e31","e32","e33","e34","e35","e36","e37","e38","e39","e40"
         ]])
# predict

model, normal_loss = train_auto_encoder(x_train[np.array(y_train) == 0])


test_tensor = torch.tensor(x_test).float()
train_tensor = torch.tensor(x_train).float()

train_encode= model.encode_linear(train_tensor).detach().numpy()
test_encode= model.encode_linear(test_tensor).detach().numpy()


#gnb = GaussianNB()
gnb = BernoulliNB(alpha=1.0, fit_prior=True,class_prior=None)
'''
used_features =[
    0,1
    ]
    '''
# train
E_train=pd.DataFrame(train_encode)
E_test=pd.DataFrame(test_encode)
gnb.fit(
    E_train[[0]].values,
    X_train["e41"]
)

# inference
y_pred = gnb.predict(E_test[[0]])

y_pred_pd=pd.DataFrame(y_pred)
y_predict=[]
# write
for i in range(y_pred_pd.shape[0]):
    if y_pred_pd.at[i,0] == 0:
        y_predict.append('normal.')
    else:
        y_predict.append('attack.')
y_pred_write=pd.DataFrame(y_predict)
X_test["Predict"]=y_pred_write
# calculate results
False_Negatives= ((X_test["e41"] >= 0.5) & (y_pred < 0.5)).sum()
False_Positives= ((X_test["e41"] < 0.5) & (y_pred >= 0.5)).sum()
True_Positives = ((X_test["e41"] >= 0.5) & (y_pred >= 0.5)).sum()
# output

print("Total {} points \nFalse Negatives={}\nFalse Positives={}\n True Positives={}\nPrecision={}\nRecall={}\nF1 Score={}"
      .format(
           X_test.shape[0],
           False_Negatives,
 	       False_Positives,
	       True_Positives,
	       True_Positives/(False_Positives+True_Positives),
	       True_Positives/(False_Negatives+True_Positives),
	       2*(True_Positives/(False_Positives+True_Positives)*True_Positives/(False_Negatives+True_Positives))/(True_Positives/(False_Positives+True_Positives)+True_Positives/(False_Negatives+True_Positives))
))
