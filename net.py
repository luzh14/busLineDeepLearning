#-*- coding=utf8 -*-
from __future__ import print_function
import scipy.io as sio
from keras.datasets.cifar import load_batch
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D,LSTM
from keras.utils import np_utils
import numpy as np
import os

def create_model(data, nb_classes):
    # print("data.shape[1:]")
    # print(data.shape[1:])
    model = Sequential()
    model.add(Convolution1D(nb_filter=16,strides=1,filter_length=4,border_mode='same',input_shape=data.shape[1:]))
    model.add(Activation('linear'))
    model.add(LSTM(output_dim=128))
    model.add(Activation('linear'))
    model.add(LSTM(output_dim=256))
    model.add(Activation('linear'))
    model.add(Dense(output_dim=128))
    model.add(Activation('tanh'))
    model.add(Dense(output_dim=1))
    model.add(Activation('linear'))
# 使用RMSprop为训练时的优化函数
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    return model

def preprocess_data(path):
    nb_train_samples = 50000
    #32*32的画面，RGB通道
    x_train = np.zeros((nb_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((nb_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        #读如图片数据
        x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000: i * 10000] = labels

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_dim_ordering() == 'tf':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)

# 输入的图片的维度
img_rows, img_cols = 32, 32
# RGB图像有三个通道.
img_channels = 3

# 使用preprocess_data处理数据:
(X_train, y_train), (X_test, y_test) = preprocess_data('/Users/luzh14/PycharmProjects/cifar10CNN/cifar-10-batches-py')
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# 由于图片像素值标准化值0-1区间
X_train /= 255
X_test /= 255

# 定义网络训练时的参数
batch_size = 500 # 由于训练集有50000张图片，因此我们为使训练快一些批训练的批大小设置为500
nb_classes = 10
nb_epoch = 10
data_augmentation = True

if not data_augmentation:
    print('Not using data augmentation.')
    model = create_model(X_train, nb_classes)
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).

    #创建网络
    model = create_model(X_train, nb_classes)
    datagen.fit(X_train)


    save_fn = 'loadData.mat'
    sio.savemat(save_fn, {'X_train': X_train, 'y_train': y_train,'X_test': X_test, 'y_test': y_test})