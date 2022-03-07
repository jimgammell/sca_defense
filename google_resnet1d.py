# Note: This code is adapted from the code at https://github.com/google/scaaml/blob/master/scaaml/intro/model.py

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam


def block(x,
          filters,
          kernel_size=3,
          strides=1,
          conv_shortcut=False,
          activation='relu'):
    """Residual block with preactivation
    From: https://arxiv.org/pdf/1603.05027.pdf
    Args:
        x: input tensor.
        filters (int): filters of the bottleneck layer.
        kernel_size(int, optional): kernel size of the bottleneck layer.
        defaults to 3.
        strides (int, optional): stride of the first layer.
        defaults to 1.
        conv_shortcut (bool, optional): Use convolution shortcut if True,
        otherwise identity shortcut. Defaults to False.
        use_batchnorm (bool, optional): Use batchnormalization if True.
        Defaults to True.
        activation (str, optional): activation function. Defaults to 'relu'.
    Returns:
        Output tensor for the residual block.
    """

    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    if conv_shortcut:
        shortcut = layers.Conv1D(4 * filters, 1, strides=strides)(x)
    else:
        if strides > 1:
            shortcut = layers.MaxPooling1D(1, strides=strides)(x)
        else:
            shortcut = x

    x = layers.Conv1D(filters, 1, use_bias=False, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    x = layers.Conv1D(filters,
                      kernel_size,
                      strides=strides,
                      use_bias=False,
                      padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    x = layers.Conv1D(4 * filters, 1)(x)
    x = layers.Add()([shortcut, x])
    return x


def stack(x, filters, blocks, kernel_size=3, strides=2, activation='relu'):
    """A set of stacked residual blocks.
    Args:
        filters (int): filters of the bottleneck layer.
        blocks (int): number of conv blocks to stack.
        kernel_size(int, optional): kernel size of the bottleneck layer.
        defaults to 3.
        strides (int, optional): stride used in the last block.
        defaults to 2.
        conv_shortcut (bool, optional): Use convolution shortcut if True,
        otherwise identity shortcut. Defaults to False.
        activation (str, optional): activation function. Defaults to 'relu'.
    Returns:
        tensor:Output tensor for the stacked blocks.
  """
    x = block(x,
              filters,
              kernel_size=kernel_size,
              activation=activation,
              conv_shortcut=True)
    for i in range(2, blocks):
        x = block(x, filters, kernel_size=kernel_size, activation=activation)
    x = block(x, filters, strides=strides, activation=activation)
    return x

def PretrainedResnet1D(printl, byte, attack_point, **kwargs):
    model = tf.keras.models.load_model(
        'google_pretrained_models/stm32f415-tinyaes-cnn-v10-ap_%s-byte_%d-len_20000'%(attack_point, byte))
    return model

def Resnet1D(input_shape,
             printl,
             **kwargs):
    printl('Generating 1D Resnet model.')
    printl('\tUsing specified input shape: {}'.format(input_shape))
    if not(type(input_shape) == tuple):
        raise TypeError('input_shape must be of type {} but is of type {}'.format(tuple, type(input_shape)))
    if not(all([type(x) == int for x in input_shape])):
        raise TypeError('input_shape must contain elements of type {} but contains elements of type {}'.format(int, ', '.join([type(x) for x in input_shape if type(x) != int])))
    if 'pool_size' in kwargs:
        pool_size = kwargs['pool_size']
        printl('\tUsing specified pool size: {}'.format(pool_size))
    else:
        pool_size = 4
        printl('\tUsing default pool size: {}'.format(pool_size))
    if not(type(pool_size) == int):
        raise TypeError('pool_size must be of type {} but is of type {}'.format(int, type(pool_size)))
    if 'filters' in kwargs:
        filters = kwargs['filters']
        printl('\tUsing specified filters: {}'.format(filters))
    else:
        filters = 8
        printl('\tUsing default filters: {}'.format(filters))
    if not(type(filters) == int):
        raise TypeError('filters must be of type {} but is of type {}'.format(int, type(filters)))
    if 'block_kernel_size' in kwargs:
        block_kernel_size = kwargs['block_kernel_size']
        printl('\tUsing specified block kernel size: {}'.format(block_kernel_size))
    else:
        block_kernel_size = 3
        printl('\tUsing default block kernel size: {}'.format(block_kernel_size))
    if not(type(block_kernel_size) == int):
        raise TypeError('block_kernel_size must be of type {} but is of type {}'.format(int, type(block_kernel_size)))
    if 'activation' in kwargs:
        activation = kwargs['activation']
        printl('\tUsing specified activation: {}'.format(activation))
    else:
        activation = 'relu'
        printl('\tUsing default activation: {}'.format(activation))
    if not(type(activation) == str):
        raise TypeError('activation must be of type {} but is of type {}'.format(str, type(activation)))
    if 'dense_dropout' in kwargs:
        dense_dropout = kwargs['dense_dropout']
        printl('\tUsing specified dense dropout: {}'.format(dense_dropout))
    else:
        dense_dropout = 0.1
        printl('\tUsing default dense dropout: {}'.format(dense_dropout))
    if not(type(dense_dropout) == float):
        raise TypeError('dense_dropout must be of type {} but is of type {}'.format(float, type(dense_dropout)))
    num_blocks = []
    for idx in range(1, 5):
        if 'blocks_stack%d'%(idx) in kwargs:
            num_blocks.append(kwargs['blocks_stack%d'%(idx)])
            printl('\tUsing specified blocks stack {}: {}'.format(idx, num_blocks[-1]))
        else:
            num_blocks.append(3 if idx in [1, 4] else 4)
            printl('\tUsing default blocks stack {}: {}'.format(idx, num_blocks[-1]))
        if not(type(num_blocks[-1])== int):
            raise TypeError('blocks_stack{} must be of type {} but is of type {}'.format(idx, int, type(num_blocks[-1])))

    inputs = layers.Input(shape=(input_shape))
    x = inputs

    # stem
    x = layers.MaxPool1D(pool_size=pool_size)(x)

    # trunk: stack of residual block
    for block_idx in range(4):
        filters *= 2
        x = stack(x,
                  filters,
                  num_blocks[block_idx],
                  kernel_size=block_kernel_size,
                  activation=activation)

    # head model: dense
    x = layers.GlobalAveragePooling1D()(x)
    for _ in range(1):
        x = layers.Dropout(dense_dropout)(x)
        x = layers.Dense(256)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)

    outputs = layers.Dense(256, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name='Discriminator')

    return model
