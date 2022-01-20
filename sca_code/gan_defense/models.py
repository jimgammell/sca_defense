from tensorflow.keras import layers, Model
import tensorflow as tf

def cumulative_model(discriminator, generator, time_input=False):
    ap_inp = layers.Input(shape=(8,))
    ap = ap_inp
    trace_inp = layers.Input(shape=(20000,))
    trace = trace_inp
    if time_input:
        time_inp = layers.Input(shape=(20000,))
        time = time_inp
    
    if time_input:
        visible_trace = generator([ap, trace, time])
    else:
        visible_trace = generator([ap, trace])
    
    #visible_trace = layers.Lambda(lambda x: x-tf.math.reduce_mean(x))(visible_trace)
    #visible_trace = layers.Lambda(lambda x: x/tf.math.reduce_max(x))(visible_trace)
    
    softmax_prediction = discriminator(visible_trace)
    
    if time_input:
        cumulative_model = Model(inputs=[ap_inp, trace_inp, time_inp], outputs=softmax_prediction)
    else:
        cumulative_model = Model(inputs=[ap_inp, trace_inp], outputs=softmax_prediction, name='Cumulative_model')
    return cumulative_model