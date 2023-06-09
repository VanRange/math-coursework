import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
import pymrmr

pca = PCA(n_components=2)

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
X_train[["0","1","2","3","4","5","6","7","8","9","10",
         "11","12","13","14","15","16","17","18","19","20",
         "21","22","23","24","25","26","27","28","29","30",
         "31","32","33","34","35","36","37","38","39","40"
         ]] = X_train[[
             "0","1","2","3","4","5","6","7","8","9","10",
         "11","12","13","14","15","16","17","18","19","20",
         "21","22","23","24","25","26","27","28","29","30",
         "31","32","33","34","35","36","37","38","39","40"
         ]].apply(gle.fit_transform) 
X_test[["0","1","2","3","4","5","6","7","8","9","10",
         "11","12","13","14","15","16","17","18","19","20",
         "21","22","23","24","25","26","27","28","29","30",
         "31","32","33","34","35","36","37","38","39","40"
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
X_train["41"]=y_train_write
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
X_test["41"]=y_test_write

# mrmr

X_train.insert(0,'41',X_train.pop('41'))
res = pymrmr.mRMR(X_train, 'MIQ', 2)

plt.scatter(X_train[[res[0]]].values,X_train[[res[1]]].values,c=train_color,s=2)
plt.xlabel("x - axis")
plt.ylabel("y - axis")
plt.savefig('MRMR_train.png',bbox_inches = 'tight',dpi=300)

plt.scatter(X_test[[res[0]]].values,X_test[[res[1]]].values,c=y_color,s=2)
plt.xlabel("x - axis")
plt.ylabel("y - axis")
plt.savefig('MRMR.png',bbox_inches = 'tight',dpi=300)

gnb = BernoulliNB(alpha=1.0, fit_prior=True,class_prior=None)
used_features = res
# train
gnb.fit(
    X_train[used_features].values,
    X_train["41"]
)
# inference
y_pred = gnb.predict(X_test[used_features])
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
False_Negatives= ((X_test["41"] == 1) & (y_pred == 0)).sum()
False_Positives= ((X_test["41"] == 0) & (y_pred == 1)).sum()
True_Positives = ((X_test["41"] == y_pred) & (X_test["41"] == True)).sum()
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
