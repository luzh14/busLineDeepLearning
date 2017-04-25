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

X_train, y_train, X_test, y_test = preprocess_data('/Users/luzh14/busLineDeepLearning/mainForDishwasher.csv',256)
x, y_train, x, y_test = preprocess_data('/Users/luzh14/busLineDeepLearning/dishwhsherBool.csv',256)

# create the model
embedding_vecor_length = 100
model = Sequential()
model.add(Embedding(5000, 100, input_length=255))
model.add(Conv1D(nb_filter=32, filter_length=4, activation='linear'))
#model.add(MaxPooling1D(pool_length=2))
model.add(LSTM(128))
model.add(Dense(
        output_dim=128))
model.add(Activation("relu"))
model.add(Dropout(0.3))
model.add(Dense(
        output_dim=256))
model.add(Activation("relu"))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, nb_epoch=10, batch_size=100)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

predicted = predict_point_by_point(model, X_test)
plt.plot(predicted)
plt.plot(y_test)
plt.show()