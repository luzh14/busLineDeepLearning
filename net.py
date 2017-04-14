#-*- coding=utf8 -*-
from __future__ import print_function
import scipy.io as sio
from keras.datasets.cifar import load_batch
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D,LSTM
from dataPreprocessing import preprocess_data
import numpy as np
import os

# 使用preprocess_data处理数据:
(X_train, y_train), (X_test, y_test) = preprocess_data('loadData.mat')

X_train /= 512
X_test /= 512

batch_size=1
nb_epoch = 10
sizeX=100
sizeY=128

model = Sequential()
model.add(Dense(units=64, input_dim=128,output_dim=10))
model.add(Activation('relu'))
model.add(Dense(units=10,output_dim=1))
model.add(Activation('softmax'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)





