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
        self.block1 = ConvBlock(filters=96, kernel=(11, 11), strides=(4, 4), padding='same', batch_norm=True)
        self.block2 = ConvBlock(filters=256, kernel=(5, 5), padding='same', batch_norm=True)
        self.block3 = ConvBlock(filters=384, kernel=(3, 3), batch_norm=True)
        self.block4 = ConvBlock(filters=384, kernel=(3, 3), batch_norm=True)
        self.block5 = ConvBlock(filters=384, kernel=(3, 3), batch_norm=False)
        self.block6 = ConvBlock(filters=256, kernel=(3, 3), padding='same')
        self.pool = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.flatten = layers.Flatten()
        self.classifier = tf.keras.Sequential([
            layers.Dense(4096),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            layers.Dense(4096),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes),
        ])

    def call(self, inputs, training=False):
        x = tf.keras.Input(shape=(im_height, im_width, channels))
        x = self.block1(inputs)
        # x = layers.Activation('relu')(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                             padding='same')(x)
        # Conv block 2
        x = self.block2(x)
        x = self.pool(x)
        # Conv block 3
        x = self.block3(x)
        # Conv block 4
        x = self.block4(x)
        # Conv block 5
        x = self.block5(x)
        # Conv block 6
        x = self.block6(x)
        x = self.pool(x)
        x = self.flatten(x)
        # Fully connected layer
        x = self.classifier(x)
        x = layers.Activation('softmax')
        return x
