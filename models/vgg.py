import tensorflow as tf
from tf.keras import Model
from utils.config import NUM_CLASSES, im_height, im_width, channels
from utils.layer_blocks import ConvBlocks
layers = tf.keras.layers


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(Model):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = tf.keras.Sequential([
            layers.Dense(4096),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            layers.Dense(4096),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES)
        ])

    def call(self, inputs):
        x = tf.keras.Input(shape=(im_height, im_width, channels))
        x = self.features(inputs)
        x = layers.Flatten()
        x = self.classifier(x)
        x = layers.Activation('softmax')
        return x


def _vgg(cfg, batch_norm, **kwargs):
    model = VGG(ConvBlocks(cfg, batch_norm=batch_norm), **kwargs)
    return model


def vgg11(**kwargs):
    """VGG-11 """
    return _vgg('vgg11', 'A', False, **kwargs)


def vgg13(**kwargs):
    """VGG-13"""
    return _vgg('vgg13', 'B', False, **kwargs)


def vgg16(**kwargs):
    """VGG-16"""
    return _vgg('vgg16', 'D', False, **kwargs)


def vgg19(**kwargs):
    """VGG-19"""
    return _vgg('vgg19', 'E', False, **kwargs)


"""
def vgg11_bn(**kwargs):

    return _vgg('vgg11_bn', 'A', True, **kwargs)


def vgg13_bn(**kwargs):

    return _vgg('vgg13_bn', 'B', True, **kwargs)


def vgg19_bn(**kwargs):

    return _vgg('vgg19_bn', 'E', True, **kwargs)
"""

if __name__ == "__main__":
    model = vgg19()
    out = model(tf.ones([10, 224, 224, 3]))
    print(out)
