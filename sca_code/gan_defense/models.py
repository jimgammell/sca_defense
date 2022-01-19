from tensorflow.keras import layers, Model

def cumulative_model(discriminator, generator, time_input=False):
    ap_inp = layers.Input(shape=(8,))
    ap = ap_inp
    trace_inp = layers.Input(shape=(20000,))
    trace = trace_inp
    if time_input:
        time_inp = layers.Input(shape=(20000,))
        time = time_imp
    
    if time_input:
        visible_trace = generator([ap, trace, time])
    else:
        visible_trace = generator([ap, trace])
    softmax_prediction = discriminator(visible_trace)
    
    if time_input:
        cumulative_model = Model(inputs=[ap_inp, trace_inp, time_inp], outputs=softmax_prediction)
    else:
        cumulative_model = Model(inputs=[ap_inp, trace_inp], outputs=softmax_prediction)
    return cumulative_model