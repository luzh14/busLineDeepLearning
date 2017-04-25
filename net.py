#-*- coding=utf8 -*-
from __future__ import print_function
import scipy.io as sio
from keras.datasets.cifar import load_batch
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D,LSTM,Conv1D
from dataPreprocessing import preprocess_data
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from keras.utils.np_utils import to_categorical



# 使用preprocess_data处理数据:
X_train, y_train, X_test, y_test = preprocess_data('main.csv',256)
x, y_train, x, y_test = preprocess_data('microBool.csv',256)


def build_model(layers):
    model = Sequential()

    model.add(Conv1D(filter_length=64, nb_filter=10,
                     input_shape=(254, 1),
                     activation='relu'))
    model.add(Conv1D(filter_length=32, nb_filter=10,
                     activation='relu'))

    model.add(Conv1D(filter_length=16, nb_filter=10,
                     activation='relu'))

    model.add(LSTM(
        #input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mean_squared_logarithmic_error", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

epochs  = 1

print('> Loading data... ')


print('> Data Loaded. Compiling...')

model = build_model([1, 32, 32, 1])

model.fit(
    X_train,
    y_train,
    batch_size=500,
    nb_epoch=epochs,
    validation_split=0.02)

predicted = predict_point_by_point(model, X_test)
plt.plot(predicted)
plt.show()
plt.plot(X_test)



