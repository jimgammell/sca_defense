from tensorflow.keras import layers, Model, activations
from tensorflow.keras import backend as K
from tensorflow.math import sin
import tensorflow as tf
import numpy as np

def IdentityGenerator():
    inp = layers.Input(shape=(8,))
    trace_inp = layers.Input(shape=(20000,))
    x = inp
    trace = trace_inp
    
    generator_model = Model(inputs=[inp, trace_inp], outputs=trace, name='Generator')
    generator_model.summary()
    return generator_model

def MlpGenerator(layer_sizes=[], min_val=-1, max_val=1):
    lg_inp = layers.Input(shape=(8,))
    lg_trace_inp = layers.Input(shape=(20000,))
    
    lg_x = lg_inp
    lg_trace = lg_trace_inp
    
    for layer in layer_sizes:
        lg_x = layers.Dense(layer)(lg_x)
        lg_x = layers.ReLU()(lg_x)
    
    lg_x = layers.Dense(20000)(lg_x)
    lg_x = activations.sigmoid(lg_x)
    lg_x = layers.Lambda(lambda x: (max_val-min_val)*x+min_val)(lg_x)
    lg_output = layers.add([lg_x, lg_trace])
    
    generator_model = Model(inputs=[lg_inp, lg_trace_inp], outputs=lg_output, name='Generator')
    generator_model.summary()
    return generator_model

def LinearGenerator(min_val=-1, max_val=1):
    return MlpGenerator(layer_sizes=[], min_val=min_val, max_val=max_val)
def Mlp1Generator(min_val=-1, max_val=1):
    return MlpGenerator(layer_sizes=[64], min_val=min_val, max_val=max_val)
def Mlp3Generator(min_val=-1, max_val=1):
    return MlpGenerator(layer_sizes=[64, 64, 64], min_val=min_val, max_val=max_val)

def CnnTransposeGenerator(min_val=-1, max_val=1):
    inp = layers.Input(shape=(8, 1))
    trace_inp = layers.Input(shape=(20000,))
    
    x = inp
    trace = trace_inp
    x = layers.Conv1DTranspose(20, 3, strides=5)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1DTranspose(20, 3, strides=5)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1DTranspose(20, 3, strides=5)(x)
    x = layers.Flatten()(x)
    x = activations.sigmoid(x)
    x = layers.Lambda(lambda x: (max_val-min_val)*x+min_val)(x)
    output = layers.add([x, trace])
    
    model = Model(inputs=[inp, trace_inp], outputs=output, name='Generator')
    model.summary()
    return model

def FourierGenerator(min_val=-1, max_val=1, n_terms=100):
    inp = layers.Input(shape=(8,))
    trace_inp = layers.Input(shape=(20000,))
    x = inp
    trace = trace_inp
    
    # Coefficients
    cx = layers.Dense(512)(x)
    cx = layers.ReLU()(cx)
    c = layers.Dense(n_terms)(cx)
    c = layers.RepeatVector(20000)(c)
    
    # Frequencies
    ox = layers.Dense(512)(x)
    ox = layers.ReLU()(ox)
    o = layers.Dense(n_terms)(ox)
    o = activations.sigmoid(o)
    o = layers.Lambda(lambda x: 2*np.pi*x)(o)
    o = layers.RepeatVector(20000)(o)
    
    # Phases
    px = layers.Dense(512)(x)
    px = layers.ReLU()(px)
    p = layers.Dense(n_terms)(px)
    p = activations.sigmoid(p)
    p = layers.Lambda(lambda x: 2*np.pi*x-np.pi)(p)
    p = layers.RepeatVector(20000)(p)
    
    time_inp = layers.Input(shape=(20000,))
    time = time_inp
    time = layers.RepeatVector(100)(time)
    time = layers.Permute((2, 1))(time)
    
    time = layers.Multiply()([o, time])
    time = layers.Add()([time, p])
    series = layers.Lambda(lambda x:sin(x))(time)
    series = layers.Multiply()([c, series])
    series = layers.Lambda(lambda x: tf.math.reduce_sum(x, axis=-1))(series)
    series = activations.sigmoid(series)
    series = layers.Lambda(lambda x: (max_val-min_val)*x+min_val)(series)
    
    output = layers.Add()([series, trace])
    
    model = Model(inputs=[inp, trace_inp, time_inp], outputs=output, name='Generator')
    model.summary()
    return model
    