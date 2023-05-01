import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dropout, Conv2D, MaxPool2D, Conv2DTranspose, Concatenate, concatenate
from keras.activations import relu

from utils import config


# up-sampling block from which is the decoder constructed
def upsampling(previous_layer, to_concatenate, n_filters):
    conv2d_transpose = Conv2DTranspose(n_filters, kernel_size=config.KERNEL_SIZE, strides=2, padding="same")(previous_layer)
    concatenate = tf.keras.layers.concatenate([conv2d_transpose, to_concatenate])
    dropout = Dropout(0.5)(concatenate)
    conv2d = double_conv2d(dropout, n_filters)

    return conv2d


# down-sampling block from which is the encoder constructed
def downsampling(previous_layer, n_filters):
    conv2d = double_conv2d(previous_layer, n_filters)
    maxpool2d = MaxPool2D(config.POOL_SIZE)(conv2d)
    dropout = Dropout(0.5)(maxpool2d)

    return dropout, conv2d


def double_conv2d(previous_layer, n_filters):
    x = Conv2D(n_filters, kernel_size=config.KERNEL_SIZE, activation=relu, padding="same")(previous_layer)
    x = Conv2D(n_filters, kernel_size=config.KERNEL_SIZE, activation=relu, padding="same")(x)

    return x


def UNet():
    input_layer = Input(config.IMAGE_SHAPE)

    downsample_block_1, downsample_conv2d_1 = downsampling(input_layer, config.FILTERS_NUM)
    downsample_block_2, downsample_conv2d_2 = downsampling(downsample_block_1, config.FILTERS_NUM * 2)
    downsample_block_3, downsample_conv2d_3 = downsampling(downsample_block_2, config.FILTERS_NUM * 2 * 2)
    downsample_block_4, downsample_conv2d_4 = downsampling(downsample_block_3, config.FILTERS_NUM * 2 * 2 * 2)

    bottleneck = double_conv2d(downsample_block_4, config.FILTERS_NUM * 2 * 2 * 2 * 2)

    upsample_block_1 = upsampling(bottleneck, downsample_conv2d_4, config.FILTERS_NUM * 2 * 2 * 2)
    upsample_block_2 = upsampling(upsample_block_1, downsample_conv2d_3, config.FILTERS_NUM * 2 * 2)
    upsample_block_3 = upsampling(upsample_block_2, downsample_conv2d_2, config.FILTERS_NUM * 2)
    upsample_block_4 = upsampling(upsample_block_3, downsample_conv2d_1, config.FILTERS_NUM)

    output_layer = Conv2D(1, kernel_size=(1, 1), padding="same", activation="sigmoid", use_bias=False)(upsample_block_4)

    model = Model(input_layer, output_layer, name="UNet")

    return model








