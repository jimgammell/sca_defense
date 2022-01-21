from tensorflow.keras import layers, Model
import tensorflow as tf

def cumulative_model(discriminator, generator, time_input=False):
    ap_bin_inp = layers.Input(shape=(8,))
    ap_bin = ap_bin_inp
    trace_inp = layers.Input(shape=(20000,))
    trace = trace_inp
    if time_input:
        time_inp = layers.Input(shape=(20000,))
        time = time_inp
    
    if time_input:
        visible_trace = generator([ap_bin, trace, time])
    else:
        visible_trace = generator([ap_bin, trace])
    
    prediction = discriminator(visible_trace)
    
    if time_input:
        cumulative_model = Model(inputs=[ap_bin_inp, trace_inp, time_inp], outputs=prediction)
    else:
        cumulative_model = Model(inputs=[ap_bin_inp, trace_inp], outputs=prediction, name='Cumulative_model')
    return cumulative_model