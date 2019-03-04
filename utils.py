import math
import numpy as np
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.datasets import cifar10
import scipy.io as sio

def find_bilinear_dimensions(d):
    upper = math.floor(math.sqrt(d))
    d1 = None
    d2 = None
    for i in range(1, upper+1):
        if d%i == 0:
            d1 = i
            d2 = d/i
    return int(d1), int(d2)


def load_cifar10():
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('y_train shape:', y_train.shape)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test

'''
def load_svhn():
    num_classes = 10
    train_data = sio.loadmat('../../data/SVHN/train_32x32.mat')
    val_data = sio.loadmat('../../data/SVHN/test_32x32.mat')
    
    x_train = train_data['X'].astype('float32')/255
    x_train = x_train.transpose((3,0,1,2))
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    y_train = train_data['y']
    y_train = y_train - 1
    y_train = to_categorical(y_train, num_classes)

    x_val = val_data['X'].astype('float32')/255
    x_val = x_val.transpose((3,0,1,2))
    x_val -= x_train_mean
    y_val = val_data['y']
    y_val = y_val - 1
    y_val = to_categorical(y_val, num_classes)

    print('X shape:')
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'val samples')
    return x_train, y_train, x_val, y_val
'''