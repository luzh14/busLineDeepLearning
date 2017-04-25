from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D,LSTM,Conv1D,Reshape,Permute
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from keras.utils.np_utils import to_categorical


def preprocess_data(path,seq_len):
    f = open(path, 'rb').read()
    data = f.decode().split('\n')
    # print(data)
    sequence_length = seq_len
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row)]
    #np.random.shuffle(train)
    x_train = train[:,:-1]
    y_train = train[:,-1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    return [x_train, y_train, x_test, y_test]

#preprocess_data('/Users/luzh14/busLineDeepLearning/main.csv',128)

X_train, y_train, X_test, y_test = preprocess_data('/Users/luzh14/busLineDeepLearning/mainForDishwasher.csv',151)
x, y_train, x, y_test = preprocess_data('/Users/luzh14/busLineDeepLearning/dishwhsherBool.csv',151)


plt.plot(y_test)
plt.show()