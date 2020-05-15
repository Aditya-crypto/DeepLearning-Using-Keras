#MULTILAYER PERCEPTRON(MLP)

from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras import optimizers
from keras import backend as K
import sys
import os

path=sys.argv[1]
input_list=[]
fp=open(path)
#reading global_active_power value
for line in fp:
    ip=line.split(';')
    input_list.append(ip[2])

#removing 1st value
inp=np.array(input_list)
inputData=inp[1:]

#removing none values
processedlist = inputData[inputData != '?']

#converting String values
dataframe=pd.DataFrame(processedlist)
dataset = dataframe.values
dataset = dataset.astype('float32')
sz=dataset.shape[0]-60

#making dataframe with window size as 60
def divide_chunks(l): 
  for i in range(0, len(l)-60):  
    yield l[i:i+60]  
x = list(divide_chunks(dataset))
x=np.array(x)
x.shape=(sz,60)
y_train=x[:,59:]
y_train=y_train[1:]
y_train=np.append(y_train,1.0)
X_train=x

#Model Training
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=60))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train,y_train, epochs=25, verbose=1)

#predicting Missing Values
predlist=[]
for i in range(0,len(inputData)-60):
  if(inputData[i+60]=='?'):
    X_test=inputData[i:i+60]
    X_test=np.array(X_test)
    X_test= X_test.astype('float32')
    X_test.shape=(1,60)
    pred=model.predict(X_test, verbose=1)
    pred.shape=(1,)
    print(pred[0])
    inputData[i+60]=str(pred[0])
    predlist.append(pred)

for b in predlist:
  print(b)

