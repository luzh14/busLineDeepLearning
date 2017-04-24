# LSTM and CNN for sequence classification in the IMDB dataset
import numpy
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
from keras.layers import Convolution1D, MaxPooling1D,LSTM,Conv1D
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from keras.utils.np_utils import to_categorical
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000


def preprocess_data(path,seq_len):
    f = open(path, 'rb').read()
    data = f.decode().split('\n')
    # print(data)
    sequence_length = seq_len - 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row)]
    np.random.shuffle(train)
    x_train = train[:,:-1]
    y_train = train[:,-1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    return [x_train, y_train, x_test, y_test]

#preprocess_data('/Users/luzh14/busLineDeepLearning/main.csv',128)

X_train, y_train, X_test, y_test = preprocess_data('/Users/luzh14/busLineDeepLearning/main.csv',256)
x, y_train, x, y_test = preprocess_data('/Users/luzh14/busLineDeepLearning/microBool.csv',256)

# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(9999, embedding_vecor_length, input_length=254))
model.add(Conv1D(nb_filter=32, filter_length=3, activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, nb_epoch=3, batch_size=500)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))