import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
from scaaml.model import load_model_from_disk
import tensorflow as tf
from tensorflow.keras import layers, Model

def PretrainedDiscriminator(byte, attack_pt):
    if not((type(byte)==int) and (0 <= byte < 16)):
        raise ValueError("Invalid byte: {}. Must be integer in [0, 15].".format(byte))
    if not(attack_pt in ['key', 'sub_bytes_in', 'sub_bytes_out']):
        raise ValueError("Invalid attack point: {}. Must be one of \"key\", \"sub_bytes_in\", \"sub_bytes_out\".".format(attack_pt))
    model = load_model_from_disk(
        'models/stm32f415-tinyaes-cnn-v10-ap_%s-byte_%d-len_20000'%(attack_pt, byte))
    return model

def NormalizedDiscriminator(byte, attack_pt):
    discriminator = PretrainedDiscriminator(byte, attack_pt)
    
    trace_inp = layers.Input(shape=(20000,))
    trace = trace_inp
    ap_inp = layers.Input(shape=(256,))
    ap = ap_inp
    
    trace = layers.Lambda(lambda x: x-tf.math.reduce_mean(x))(trace)
    trace = layers.Lambda(lambda x: x/tf.math.reduce_std(x))(trace)
    
    output = discriminator(trace)
    
    model = Model(inputs=trace_inp, outputs=output, name='Discriminator')
    return model