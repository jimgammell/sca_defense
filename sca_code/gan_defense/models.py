from tensorflow.keras import layers, Model
import tensorflow as tf

class GanModel(Model):
    def __init__(self, disc_cost, gen_cost, *args):
        super().__init__(args)
        self.training_disc = False
        self.training_gen = False
        self.disc_cost = disc_cost
        self.gen_cost = gen_cost
        self.gen_loss_list = []
    def gen_step(self, data):
        x, y = data
        disc_vars = self.get_layer(name='Discriminator').trainable_variables
        gen_vars = self.get_layer(name='Generator').trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as gtape:
            with tf.GradientTape(watch_accessed_variables=False) as dtape:
                dtape.watch(disc_vars)
                gtape.watch(disc_vars)
                gtape.watch(gen_vars)
                pred = self(x, training=True)
                dloss = self.disc_cost(y, pred)
            d_grads = dtape.gradient(dloss, disc_vars)
            gloss = self.gen_cost(y, pred, d_grads)
        gradients = gtape.gradient(gloss, gen_vars)
        self.optimizer.apply_gradients(zip(gradients, gen_vars))
        self.compiled_metrics.update_state(y, y_pred)
        self.gen_loss_list.append(gloss)
        return {m.name: m.result() for m in self.metrics}
    def disc_step(self, data):
        x, y = data
        disc_vars = self.get_layer(name='Discriminator').trainable_variables
        with tf.GradientTape() as dtape:
            dtape.watch(disc_vars)
            pred = self(x, training=True)
            dloss = self.disc_cost(y, pred)
        d_grads = dtape.gradient(dloss, disc_vars)
        gloss = self.gen_cost(y, pred, d_grads)
        self.optimizer.apply_gradients(zip(gradients, disc_vars))
        self.compiled_metrics.update_state(y, y_pred)
        self.gen_loss_list.append(gloss) 
        return {m.name: m.result() for m in self.metrics}
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
        cumulative_model = GanModel(disc_cost, gen_cost, inputs=[ap_bin_inp, trace_inp, time_inp], outputs=prediction)
    else:
        cumulative_model = GanModel(disc_cost, gen_cost, inputs=[ap_bin_inp, trace_inp], outputs=prediction, name='Cumulative_model')
    return cumulative_model