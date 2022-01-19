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
    
    generator_model = Model(inputs=[inp, trace_inp], outputs=trace)
    generator_model.summary()
    return generator_model

def MlpGenerator(layers=[], min_val=-1, max_val=1):
    lg_inp = layers.Input(shape=(8,))
    lg_trace_inp = layers.Input(shape=(20000,))
    
    lg_x = lg_inp
    lg_trace = lg_trace_inp
    
    for layer in layers:
        lg_x = layers.Dense(layer)(lg_x)
        lg_x = layers.ReLU()(lg_x)
    
    lg_x = layers.Dense(20000)(lg_x)
    lg_x = (max_val-min_val)*activations.sigmoid(lg_x)+min_val
    lg_output = layers.add([lg_x, lg_trace])
    lg_output *= .5
    
    generator_model = Model(inputs=[lg_inp, lg_trace_inp], outputs=lg_output, name='Generator')
    generator_model.summary()
    return generator_model

def LinearGenerator(min_val=-1, max_val=1):
    return MlpGenerator(layers=[], min_val=min_val, max_val=max_val)
def Mlp1Generator(min_val=-1, max_val=1):
    return MLPGenerator(layers=[64], min_val=min_val, max_val=max_val)
def Mlp3Generator(min_val=-1, max_val=1):
    return MLPGenerator(layers=[64, 64, 64], min_val=min_val, max_val=max_val)

def CnnTransposeGenerator(min_val=-1, max_val=1):
    inp = layers.Input(shape=(8, 1))
    trace_inp = layers.Input(shape=(20000,))
    
    x = inp
    trace = trace_inp
    x = layers.Conv1DTranspose(8, 3, strides=4)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1DTranspose(8, 3, strides=4)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1DTranspose(8, 3, strides=4)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(20000)(x)
    x = (max_val-min_val)*activations.sigmoid(x)+min_val
    output = layers.add([x, trace])   
    output *= .5
    
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
    cx = layers.Dense(n_terms)(cx)
    c = (max_val-min_val)*activations.sigmoid(cx)+min_val
    c = layers.RepeatVector(20000)(c)
    
    # Frequencies
    ox = layers.Dense(512)(x)
    ox = layers.ReLU()(ox)
    o = layers.Dense(n_terms)(ox)
    o = 2*np.pi*activations.sigmoid(o)
    o = layers.RepeatVector(20000)(o)
    
    # Phases
    px = layers.Dense(512)(x)
    px = layers.ReLU()(px)
    p = layers.Dense(n_terms)(px)
    p = 2*np.pi*activations.sigmoid(p)-np.pi
    p = layers.RepeatVector(20000)(p)
    
    time_inp = layers.Input(shape=(20000,))
    time = time_inp
    time = layers.RepeatVector(100)(time)
    time = tf.squeeze(time)
    
    output = c*layers.Lambda(lambda x:sin(x))(o*time+p)
    output = tf.math.reduce_sum(output, axis=-1)
    output *= .5
    
    model = Model(inputs=[inp, trace_inp, time_inp], outputs=output, name='Generator')
    model.summary()
    return model
    