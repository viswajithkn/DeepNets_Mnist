# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 17:27:58 2018

@author: karavi01
"""

import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Conv2D,MaxPooling2D,Flatten,Input,BatchNormalization,AveragePooling2D,LeakyReLU
from keras.utils import plot_model
from keras import optimizers
from keras.models import Model


def inceptionKeras(input_img,numFilters11_1,numFilters11_2,numFilters11_3,numFilters33_1,numFilters55_1,numFilters_pool):
    tower_11_1 = Conv2D(numFilters11_1, (1,1), padding='same', activation='relu')(input_img)
    tower_11_2 = Conv2D(numFilters11_2, (1,1), padding='same', activation='relu')(input_img)
    tower_33_1 = Conv2D(numFilters33_1, (3,3), padding='same', activation='relu')(tower_11_2)
    tower_11_3 = Conv2D(numFilters11_3, (1,1), padding='same', activation='relu')(input_img)
    tower_55_1 = Conv2D(numFilters55_1, (5,5), padding='same', activation='relu')(tower_11_3)    
    tower_33_pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
    tower_33_pool = Conv2D(numFilters_pool, (1,1), padding='same', activation='relu')(tower_33_pool)
    
    output = keras.layers.concatenate([tower_11_1, tower_33_1, tower_55_1, tower_33_pool], axis = 3)    
    output = Activation('relu')(output)
    return output


trainFilePath = 'C:/Users/karavi01/Documents/PersonalDocs/PythonKaggle/mnist/train.csv'
testFilePath = 'C:/Users/karavi01/Documents/PersonalDocs/PythonKaggle/mnist/test.csv'
modelType = 'inception'
rawTrainData = pd.read_csv(trainFilePath)
trainLabels = rawTrainData['label']
trainData = rawTrainData.iloc[:,1:42000]

rawTestData = pd.read_csv(testFilePath)
testData = rawTestData.values


tempArray = trainData.iloc[6,:].values
tempArray = np.reshape(tempArray,(28,28))
plt.pyplot.imshow(tempArray)
plt.pyplot.title(trainLabels.iloc[6])

tempArray = trainData.iloc[46,:].values
tempArray = np.reshape(tempArray,(28,28))
plt.pyplot.imshow(tempArray)
plt.pyplot.title(trainLabels.iloc[46])

Y = keras.utils.to_categorical(trainLabels,num_classes = 10)
X = trainData.values
x_train,x_val,y_train,y_val = train_test_split(X,Y,test_size = 0.25, random_state = 42)

if modelType == 'regular':
    model = Sequential()
    model.add(Dense(300,activation='relu',input_dim = 784))
    model.add(Dropout(0.5))
    model.add(Dense(100,activation='tanh'))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(10,activation='softmax'))
    model.compile(optimizer='Adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
    #SGDOpt = optimizers.SGD(lr=0.01, momentum=0.01, decay=0.01, nesterov=False)
    #model.compile(optimizer='SGD',loss = 'categorical_crossentropy',metrics=['accuracy'])
    model.fit(x=x_train,y=y_train,batch_size = 32,epochs = 100)
    score = model.evaluate(x_val,y_val, batch_size = 32)
    print('The accuracy of a 3 layer sequential model on mnist data is: ',score[1])

x_train = np.reshape(x_train,(31500,28,28,1))
x_val = np.reshape(x_val,(10500,28,28,1))
input_img = Input(shape=(28,28,1))
testData = np.reshape(testData,(28000,28,28,1))

if modelType == 'lenet':
    model_conv = Sequential()
    model_conv.add(Conv2D(filters = 6,kernel_size = (5,5),strides = 1,padding = 'same'))
    model_conv.add(Activation('relu'))
    model_conv.add(MaxPooling2D(pool_size = (2,2),padding = 'same'))
    model_conv.add(Dropout(0.5))
    model_conv.add(Conv2D(filters = 16,kernel_size = (5,5),strides = 1,padding = 'same'))
    model_conv.add(Activation('relu'))
    model_conv.add(MaxPooling2D(pool_size = (2,2),padding = 'same'))
    #model_conv.add(Dropout(0.5))
    model_conv.add(Conv2D(filters = 120,kernel_size = (5,5),strides = 1,padding = 'same'))
    model_conv.add(Activation('relu'))
    model_conv.add(Flatten())
    model_conv.add(Dense(84,activation='relu'))
    model_conv.add(Dense(10,activation='softmax'))
    model_conv.compile(optimizer='Adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
    model_conv.fit(x=x_train,y=y_train,batch_size = 32,epochs = 100)
    score = model_conv.evaluate(x_val,y_val, batch_size = 64)
    print('The accuracy of a LeNet 5 on mnist data is: ',score[1])
    
    
    predictedProb = model_conv.predict(testData,batch_size=32)
    results = np.argmax(predictedProb,axis = 1)
    results = pd.Series(results,name="Label")
    submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
    submission.to_csv("LeNet5.csv",index=False)

if modelType == 'inception':
    layer_1 = Conv2D(filters = 64,kernel_size = (5,5),strides = 1,padding = 'same',activation='relu')(input_img)
    layer_2 = Conv2D(filters = 128,kernel_size = (3,3),strides = 1,padding = 'same',activation='relu')(layer_1)
    layer_2 = MaxPooling2D(pool_size = (3,3),padding = 'same',strides = 2)(layer_2)
    layer_2 = BatchNormalization(axis=3, momentum=0.99, epsilon=0.001)(layer_2)
    layer_incp_1 = inceptionKeras(layer_2,8,64,8,96,16,8)
    layer_incp_2 = inceptionKeras(layer_incp_1,64,96,16,128,32,32)
    layer_incp_3 = inceptionKeras(layer_incp_2,160,112,24,224,64,64)
    layer_3 = MaxPooling2D(pool_size = (3,3),padding = 'same',strides = 2)(layer_incp_3)
    layer_3 = BatchNormalization()(layer_3)
    layer_incp_4 = inceptionKeras(layer_3,128,128,32,256,64,64)
    layer_4 = AveragePooling2D(pool_size = (7,7),padding = 'same',strides = 7)(layer_incp_4)
    output = Flatten()(layer_4)
    output = Dropout(0.5)(output)
    output = Dense(512,activation='relu')(output)
    out = Dense(10, activation='softmax')(output)
    
    model_conv = Model(inputs = input_img, outputs = out)
    model_conv.compile(optimizer='Adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
    model_conv.fit(x_train, y_train, batch_size=32,epochs = 10)
    score = model_conv.evaluate(x_val,y_val, batch_size = 64)
    print('The accuracy of a Inception 1 on mnist data is: ',score[1])
    
    predictedProb = model_conv.predict(testData,batch_size=32)
    results = np.argmax(predictedProb,axis = 1)
    results = pd.Series(results,name="Label")
    submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
    submission.to_csv("Inception.csv",index=False)