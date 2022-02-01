from tensorflow.keras import layers, Model
import tensorflow as tf

class GanModel(Model):
    def __init__(self, disc_cost, gen_cost, **kwargs):
        super(GanModel, self).__init__(**kwargs)
        self.training_disc = False
        self.training_gen = False
        self.disc_cost = disc_cost
        self.gen_cost = gen_cost
    def gen_step(self, data):
        x, y = data
        disc_vars = self.get_layer(name='Discriminator').get_layer(name='model').get_layer(index=-1).trainable_variables
        gen_vars = self.get_layer(name='Generator').trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as gtape:
            with tf.GradientTape(watch_accessed_variables=False) as dtape:
                dtape.watch(disc_vars)
                gtape.watch(disc_vars)
                gtape.watch(gen_vars)
                pred = self(x, training=True)
                dloss = self.disc_cost(y, pred)
            d_grads = dtape.gradient(dloss, disc_vars)
            d_grads = tf.concat([tf.reshape(t, [-1]) for t in d_grads], axis=0)
            gloss = self.gen_cost(y, pred, d_grads)
        gradients = gtape.gradient(gloss, gen_vars)
        self.optimizer.apply_gradients(zip(gradients, gen_vars))
        self.compiled_metrics.update_state(y, pred)
        otp = {m.name: m.result() for m in self.metrics}
        otp.update({'gen_cost': gloss})
        return otp
    def disc_step(self, data):
        x, y = data
        disc_vars = self.get_layer(name='Discriminator').trainable_variables        
        disc_otp_vars = self.get_layer(name='Discriminator').get_layer(name='model').get_layer(index=-1).trainable_variables
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as dtape:
            dtape.watch(disc_vars)
            pred = self(x, training=True)
            dloss = self.disc_cost(y, pred)
        d_grads = dtape.gradient(dloss, disc_vars)
        d_grads_otp = dtape.gradient(dloss, disc_otp_vars)
        d_grads_otp = tf.concat([tf.reshape(t, [-1]) for t in d_grads_otp], axis=0)
        gloss = self.gen_cost(y, pred, d_grads_otp)
        self.optimizer.apply_gradients(zip(d_grads, disc_vars))
        self.compiled_metrics.update_state(y, pred)
        otp = {m.name: m.result() for m in self.metrics}
        otp.update({'gen_cost': gloss})
        return otp
    def train_step(self, data):
        if self.training_disc == True:
            out = self.disc_step(data)
        elif self.training_gen == True:
            out = self.gen_step(data)
        else:
            assert False
            out = None
        return out

def cumulative_model(discriminator, generator, disc_cost, gen_cost, time_input=False):
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
        cumulative_model = GanModel(disc_cost, gen_cost, inputs=[ap_bin_inp, trace_inp, time_inp], outputs=prediction, name='Cumulative_model')
    else:
        cumulative_model = GanModel(disc_cost, gen_cost, inputs=[ap_bin_inp, trace_inp], outputs=prediction, name='Cumulative_model')
    return cumulative_model