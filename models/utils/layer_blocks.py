import tensorflow as tf
from tensorflow.keras.layers import Layer
layers = tf.keras.layers


class ConvBlocks(Layer):
    '''
    This class defines a block generator for 2d Convolution operation with BN
    and activation
    -------------------------------------------------------------------------
    # Parameters
        # init
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window. Default=(3,3)
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the width and
                height. Can be a single integer to specify the same value for
                all spatial dimensions. Default=1
            nl: String, nonlinearity activation type. Default='RE'
            channel_axis: 1 if channels are first in the image and -1 if the
                last. Default=-1
            padding_scheme: Padding scheme to apply for convolution.
                Default='same'
        # call
            x: Tensor, input tensor of conv layer.
            training: Mode for training-aware layers
    # Returns
        Output tensor.
    '''

    def __init__(self, cfg, batch_norm=False):
        super(ConvBlocks, self).__init__()
        self.cfg = cfg
        self.batch_norm = batch_norm

    def call(self, inputs):
        layer = []
        for v in self.cfg:
            if v == 'M':
                layer += [layers.MaxPool2D(pool_size=(2, 2), strides=2,
                                           padding='valid')]
            else:
                if self.batch_norm:
                    layer += [ConvBlock(v, kernel=(3, 3), padding='same',
                                        batch_norm=self.batch_norm,
                                        activation='relu')]
                else:
                    layer += [ConvBlock(v, kernel=(3, 3), padding='same',
                                        activation='relu')]
        return tf.keras.Sequential(layer)


class ConvBlock(Layer):
    '''
    Convolution Block
    This class defines a 2D convolution operation with BN and activation.
    ---------------------------------------------------------------------
    # Parameters
        # init
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window. Default=(3,3)
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the width and
                height. Can be a single integer to specify the same value for
                all spatial dimensions. Default=1
            nl: String, nonlinearity activation type. Default='RE'
            channel_axis: 1 if channels are first in the image and -1 if the
                last. Default=-1
            padding_scheme: Padding scheme to apply for convolution.
                Default='same'
        # call
            x: Tensor, input tensor of conv layer.
            training: Mode for training-aware layers
    # Returns
        Output tensor.
    '''

    def __init__(self, filters, kernel=(3, 3), strides=1,
                 padding='same', channel_axis=-1, batch_norm=False,
                 activation='relu'):
        super(ConvBlock, self).__init__()
        self.channel_axis = channel_axis
        self.batch_norm = batch_norm
        self.conv = layers.Conv2D(filters, kernel, padding=padding,
                                  strides=strides)
        self.bn = layers.BatchNormalization(axis=channel_axis)
        self.activate = activation

    def call(self, inputs, training=True):
        x = self.conv(inputs)
        if self.batch_norm:
            x = self.bn(x)
        x = layers.Activation(self.activate)(x)
        return x
