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
import matplotlib.pyplot as plt
import time
import os

# 使用preprocess_data处理数据:
X_train, y_train, X_test, y_test = preprocess_data('main.csv',128)
NULL, y_train, NULL, y_test = preprocess_data('microwave.csv',128)

X_train=X_train.reshape(26887,126)

def build_model(layers):
    model = Sequential()
    model.add(Dense(1024, input_dim=126))
    model.add(Activation("sigmoid"))

    model.add(Dense(
        output_dim=1024))
    model.add(Activation("sigmoid"))

    model.add(Dense(
        output_dim=1024))
    model.add(Activation("sigmoid"))

    model.add(Dense(
        output_dim=1))
    model.add(Activation("sigmoid"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

epochs  = 10

print('> Loading data... ')


print('> Data Loaded. Compiling...')

model = build_model([1, 128, 100, 1])

model.fit(
    X_train,
    y_train,
    batch_size=250,
    nb_epoch=epochs,
    validation_split=0.05)

X_test=X_test.reshape(2987,126)
predicted = predict_point_by_point(model, X_test)
plt.figure(1)
plt.plot(predicted)
plt.show()
plt.figure(2)
plt.plot(y_test)






