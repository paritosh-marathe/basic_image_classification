import tensorflow as tf
from utils.config import NUM_CLASSES, im_height, im_width, channels
from utils.layer_blocks import ConvBlock
layers = tf.keras.layers


class Alexnet(tf.keras.Model):
    '''
    Alexnet
    Neural network model consisting of layers propsed by AlexNet paper.
    ref :  "http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf"
    ------------------------------------------------------------------------
    input_shape : 244, 244, 3

    Params:
    num_classes : input number classes to be classified
    '''

    def __init__(self):
        super(Alexnet, self).__init__()
        self.num_classes = NUM_CLASSES

    def call(self, input):
        x = layers.Conv2D(filters=96, input_shape=(im_height, im_width, channels),
                          kernel_size=(11, 11), strides=(4, 4),
                          padding='same')(input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                             padding='same')(x)
        # Conv block 2
        x = ConvBlock(filters=256, kernel=(5, 5), padding='same',
                      batch_norm=True)(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        # Conv block 3
        x = ConvBlock(filters=384, kernel=(3, 3), batch_norm=True)(x)
        # Conv block 4
        x = ConvBlock(filters=384, kernel=(3, 3), batch_norm=True)(x)
        # Conv block 5
        x = ConvBlock(filters=384, kernel=(3, 3), batch_norm=False)(x)
        x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                          padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = layers.Flatten()(x)
        # Fully connected layer
        x = layers.Dense(4096)(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(4096, activation='relu')(x)
        x = layers.Activation('relu')(x)
        x = layers.Dense(self.num_classes)(x)
        x = layers.Activation('softmax')(x)
        return x
