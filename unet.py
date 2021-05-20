from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Concatenate, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
# from keras.utils.vis_utils import plot_model

import numpy as np

"""
activation relu is done before the batchnorm as to allow the batchnorm to directly impact the next layer
instead of first doing a batchnorm and then apply a relu.

we use he-init because we use relu as an activation function
use biases is also set to true
padding is same
blocks is used for the number of levels
"""


def Encoding(inputs, filters=32, blocks=4):
    outputs = []
    x = inputs
    for index in range(blocks):
        
        x = Conv2D(filters * np.power(2, index), kernel_size=(3, 3), strides=(1, 1), use_bias=True, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters * np.power(2, index), kernel_size=(3, 3), strides=(1, 1), use_bias=True, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        if index != blocks - 1:
            outputs.append(x)
            x = MaxPooling2D((2, 2))(x)

    return x, outputs


def Decoding(inputs_1, inputs_2, filters=32, blocks=4):
    x = inputs_1
    for index in np.arange(blocks - 2, -1, -1):
        
        x = Conv2DTranspose(filters * np.power(2, index), kernel_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = Concatenate(axis=3)([x, inputs_2[index]])

        x = Conv2D(filters * np.power(2, index), kernel_size=(3, 3), strides=(1, 1), use_bias=True, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters * np.power(2, index), kernel_size=(3, 3), strides=(1, 1), use_bias=True, padding='same', kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

    return x


def UNet(input_shape, filters=32, blocks=4, channel=3):
    inputs = Input(input_shape)

    x1, EncodingList = Encoding(inputs, filters, blocks)
    
    x2 = Decoding(x1, EncodingList, filters, blocks)

    outputs = Conv2D(channel, (1, 1), activation='softmax')(x2)

    model = Model([inputs], [outputs])
    return model


def TestModel():
    features = 48
    blocks = 5
    patchsize = 128 # for patch size try 512, 256

    model = UNet((patchsize,patchsize,3), filters=features, blocks = blocks)
    # model.count_params()
    model.summary()
    # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
# TestModel()

