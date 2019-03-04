import numpy as np
from tensorflow.python.keras.layers import AveragePooling2D, Input, Dense, Flatten, Reshape
from tensorflow.python.keras.layers.convolutional import Conv2D
from LiteConv2D import BiConv2D
from LiteDense import BiDense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.python.keras.utils import multi_gpu_model
from utils import load_cifar10, load_svhn


def SVGG(input_shape=(32,32,3), n_cls = 10):
    """ A small VGG net
    # Arguments
        input_shape (tensor): shape of input image tensor
        n_cls (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    img_input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = Conv2D(32, (3, 3), strides=(2,2), activation='relu', padding='same', name='block1_conv3')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = Conv2D(64, (3, 3), strides=(2,2), activation='relu', padding='same', name='block2_conv3')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(128, (3, 3), strides=(2,2), activation='relu', padding='same', name='block3_conv3')(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(n_cls, activation='softmax')(x)
    return Model(img_input, x)

def BiSVGG(input_shape=(32,32,3), n_cls = 10, scale=1):
    """ A small VGG net with bilinear projection
    # Arguments
        input_shape (tensor): shape of input image tensor
        n_cls (int): number of classes (CIFAR10 has 10)
        scale (int): the scaling parameter for bilinear mappings
    # Returns
        model (Model): Keras model instance
    """
    img_input = Input(shape=input_shape)
    x = BiConv2D(32, (3, 3), activation='relu', padding='same', scale=scale, name='block1_conv1')(img_input)
    x = BiConv2D(32, (3, 3), activation='relu', padding='same', scale=scale, name='block1_conv2')(x)
    x = BiConv2D(32, (3, 3), strides=(2,2), activation='relu', scale=scale, padding='same', name='block1_conv3')(x)
    x = BiConv2D(64, (3, 3), activation='relu', padding='same', scale=scale, name='block2_conv1')(x)
    x = BiConv2D(64, (3, 3), activation='relu', padding='same', scale=scale, name='block2_conv2')(x)
    x = BiConv2D(64, (3, 3), strides=(2,2), activation='relu', padding='same', scale=scale, name='block2_conv3')(x)
    x = BiConv2D(128, (3, 3), activation='relu', padding='same', scale=scale, name='block3_conv1')(x)
    x = BiConv2D(128, (3, 3), activation='relu', padding='same', scale=scale, name='block3_conv2')(x)
    x = BiConv2D(128, (3, 3), strides=(2,2), activation='relu', padding='same', scale=scale, name='block3_conv3')(x)
    x = Flatten()(x)
    x = BiDense(1024*scale, activation='relu')(x)
    x = Dense(n_cls, activation='softmax')(x)
    return Model(img_input, x)

def cifar10_train_svgg():
    x_train, y_train, x_test, y_test = load_cifar10()
    model = SVGG()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test))

def cifar10_train_bisvgg():
    x_train, y_train, x_test, y_test = load_cifar10()
    model = BiSVGG(scale=1)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test))
    



if __name__ == '__main__':
    cifar10_train_bisvgg()

    
    

    