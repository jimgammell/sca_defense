from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from scaaml.utils import get_num_gpu


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


def Discriminator(trace_shape, mdl_cfg, optim_cfg):

    pool_size = mdl_cfg['initial_pool_size']
    filters = mdl_cfg['initial_filters']
    block_kernel_size = mdl_cfg['block_kernel_size']
    activation = mdl_cfg['activation']
    dense_dropout = mdl_cfg['dense_dropout']
    num_blocks = [
        mdl_cfg['blocks_stack1'], mdl_cfg['blocks_stack2'],
        mdl_cfg['blocks_stack3'], mdl_cfg['blocks_stack4']
    ]

    trace = layers.Input(shape=trace_shape)
    x = trace

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
        
        
    ap_candidate = layers.Input(shape=(8,)) # One input per bit
    k = ap_candidate
    k = layers.Dense(256)(k)
    k = layers.Activation(activation)(k)
    k = layers.Dense(1024)(k)
    k = layers.Activation(activation)(k)
    k = layers.Dense(256)(k)
    k = layers.Activation(activation)(k)
    
    pred = layers.Concatenate()([x, k])
    pred = layers.Dense(1024)(pred)
    pred = layers.Activation(activation)(pred)
    pred = layers.Dense(256)(pred)
    pred = layers.Activation(activation)(pred)
    pred = layers.Dense(1)(pred)
    pred = layers.Activation('sigmoid')(pred)

    model = Model(inputs=[trace, ap_candidate], outputs=pred)
    model.summary()

    if get_num_gpu() > 1:
        lr = optim_cfg['multi_gpu_lr']
    else:
        lr = optim_cfg['lr']

    model.compile(loss=["categorical_crossentropy"],
                  metrics=['acc'],
                  optimizer=Adam(lr))
    return model

def Generator(trace_shape):    
    attack_point = layers.Input(shape=(8,))
    x = attack_point
    
    x = layers.Dense(1000)(x)
    x = layers.Activation('relu')(x)
    additive_noise = layers.Dense(int(trace_shape[0]))(x)
    additive_noise = layers.Reshape(trace_shape)(additive_noise)
    
    trace = layers.Input(shape=trace_shape)
    
    visible_trace = layers.Add()([trace, additive_noise])
    
    model = Model(inputs=[attack_point, trace], outputs=visible_trace)
    model.summary()
    
    return model
    
    
    
    
    
    
    