# CONVOLUTIONAL NEURAL NETWORK


#Importing Libraries
from __future__ import print_function
import sys
import os
import keras
from keras.datasets import mnist
from mnist import MNIST
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

#reading MNIST DATA
mndata = MNIST(sys.argv[1])
mndata.gz = True
x_train,y_train = mndata.load_training()
x_train=np.asarray(x_train)
train_y=np.asarray(y_train)
x_test,y_test=mndata.load_testing()
x_test=np.asarray(x_test)
test_y=np.asarray(y_test)

#Reshaping Train and test data
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

#converting into categorical data
train_y = keras.utils.to_categorical(train_y,10)
test_y = keras.utils.to_categorical(test_y,10)

#Building CNN
model=Sequential()
model.add(Conv2D(32,5,data_format='channels_last',activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64,5,data_format='channels_last',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#dropping weights(image compression)
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
#Compiling our model with defined layers and attributes
model.compile(optimizer="SGD",loss="categorical_crossentropy",metrics=["accuracy"])
model.fit(x_train,train_y,batch_size=64,epochs=20,validation_data=(x_test, test_y))

predict=model_cnn.predict_classes(x_test,verbose=1)
print(predict)
