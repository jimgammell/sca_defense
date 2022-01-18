from tensorflow.keras import layers, Model, activations

def LinearGenerator(min_val, max_val):
    pass

def CnnTransposeGenerator(min_val, max_val):
    x = layers.Input(shape=(256,))
    x = layers.Conv1DTranspose(8, 3, strides=.25)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1DTranspose(8, 3, strides=.25)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1DTranspose(8, 3, strides=.25)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(20000)(x)
    output = (max_val-min_val)*activations.sigmoid(x)+min_val
    return output

def FourierGenerator(min_val, max_val):
    pass