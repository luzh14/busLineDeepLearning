#-*- coding=utf8 -*-
import scipy.io as scio
import numpy as np
import os

def preprocess_data(path):
    dataset = scio.loadmat(path)

    x_train = dataset['X']
    y_train = dataset['Y']


    x_test = dataset['X']
    y_test = dataset['Y']

    return (x_train, y_train), (x_test, y_test)
