import os
import time
import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras

import utils
import google_resnet1d
import datasets

def printl(s=''):
    utils.printl('(models):'.ljust(utils.get_pad_width()) + s)

class Generator(keras.Sequential):
    def __init__(self, model_layers, key):
        super(Generator, self).__init__(model_layers, name='Generator_%s'%(hex(key)))
        self.key = key

def one_norm_loss():
    def loss(ground_truth, prediction):
        return -tf.norm(prediction, ord=1)
    return loss

class GAN:
    valid_optimizers = {
        'SGD': keras.optimizers.SGD,
        'RMSprop': keras.optimizers.RMSprop,
        'Adam': keras.optimizers.Adam}
    valid_losses = {
        'CategoricalCrossentropy': keras.losses.CategoricalCrossentropy,
        'one_norm': one_norm_loss}
    def __init__(self, 
                 generators,
                 discriminator,
                 gen_optimizer = None,
                 gen_optimizer_kwargs = None,
                 gen_loss = None,
                 gen_loss_kwargs = None,
                 disc_optimizer = None,
                 disc_optimizer_kwargs = None,
                 disc_loss = None,
                 disc_loss_kwargs = None):
        printl('Generating full GAN model.')
        if not(type(generators) == dict):
            raise TypeError('generators must be of type {} but is of type {}'.format(dict, type(generators)))
        if not(all([isinstance(generators[k], Generator) for k in generators])):
            raise TypeError('all elements in generators must be instance of class {}'.format(Generator))
        if not(isinstance(discriminator, keras.Model)):
            raise TypeError('discriminator must be instance of class {}'.format(Discriminator))
        if gen_optimizer == None:
            gen_optimizer = keras.optimizers.Adam
            printl('\tUsing default generator optimizer: {}'.format(gen_optimizer))
        elif gen_optimizer in self.valid_optimizers:
            gen_optimizer = self.valid_optimizers[gen_optimizer]
            printl('\tUsing specified generator optimizer: {}'.format(gen_optimizer))
        else:
            raise ValueError('Invalid gen_optimizer: {}'.format(gen_optimizer))
        if gen_optimizer_kwargs == None:
            gen_optimizer_kwargs = {}
            printl('\tUsing no generator optimizer kwargs.')
        else:
            printl('\tUsing specified generator optimizer kwargs: {}'.format(gen_optimizer_kwargs))
        if not(type(gen_optimizer_kwargs) == dict):
            raise TypeError('gen_optimizer_kwargs must be of type {} but is of type {}'.format(dict, type(gen_optimizer_kwargs)))
        if gen_loss == None:
            gen_loss = keras.losses.CategoricalCrossentropy
            printl('\tUsing default generator loss: {}'.format(gen_loss))
        elif gen_loss in self.valid_losses:
            gen_loss = self.valid_losses[gen_loss]
            printl('\tUsing specified generator loss: {}'.format(gen_loss))
        else:
            raise ValueError('Invalid gen_loss: {}'.format(gen_loss))
        if gen_loss_kwargs == None:
            gen_loss_kwargs = {}
            printl('\tUsing no generator loss kwargs.')
        else:
            printl('\tUsing specified generator loss kwargs: {}'.format(gen_loss_kwargs))
        if not(type(gen_loss_kwargs) == dict):
            raise TypeError('gen_loss_kwargs must be of type {} but is of type {}'.format(dict, type(gen_loss_kwargs)))
        if disc_optimizer == None:
            disc_optimizer = keras.optimizers.Adam
            printl('\tUsing default discriminator optimizer: {}'.format(disc_optimizer))
        elif disc_optimizer in self.valid_optimizers:
            disc_optimizer = self.valid_optimizers[disc_optimizer]
            printl('\tUsing specified discriminator optimizer: {}'.format(disc_optimizer))
        else:
            raise ValueError('Invalid disc_optimizer: {}'.format(disc_optimizer))
        if disc_optimizer_kwargs == None:
            disc_optimizer_kwargs = {'lr': 0.001}
            printl('\tUsing default discriminator optimizer kwargs: {}'.format(disc_optimizer_kwargs))
        else:
            printl('\tUsing specified discriminator optimizer kwargs: {}'.format(disc_optimizer_kwargs))
        if not(type(disc_optimizer_kwargs) == dict):
            raise TypeError('disc_optimizer_kwargs must be of type {} but is of type {}'.format(dict, type(disc_optimizer_kwargs)))
        if disc_loss == None:
            disc_loss = keras.losses.CategoricalCrossentropy
            printl('\tUsing default discriminator loss: {}'.format(disc_loss))
        elif disc_loss in self.valid_losses:
            disc_loss = self.valid_losses[disc_loss]
            printl('\tUsing specified discriminator loss: {}'.format(disc_loss))
        else:
            raise ValueError('Invalid disc_loss value: {}'.format(disc_loss))
        if disc_loss_kwargs == None:
            disc_loss_kwargs = {}
            printl('\tUsing no discriminator loss kwargs')
        else:
            printl('\tUsing specified discriminator loss kwargs: {}'.format(disc_loss_kwargs))
        if not(type(disc_loss_kwargs) == dict):
            raise TypeError('disc_loss_kwargs must be of type {} but is of type {}'.format(dict, type(disc_loss_kwargs)))
            
        self.generators = generators
        self.gen_optimizer = gen_optimizer(**gen_optimizer_kwargs)
        self.gen_loss = lambda label, pred: -gen_loss(**gen_loss_kwargs)(label, pred)
        self.discriminator = discriminator
        self.disc_optimizer = disc_optimizer(**disc_optimizer_kwargs)
        self.disc_loss = disc_loss(**disc_loss_kwargs)
    def gen_train_step(self, key, batch):
        (trace, plaintext, attack_point) = batch
        trainable_vars = self.generators[key].trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(trainable_vars)
            g_trace = self.generators[key](plaintext, training=True)
            protected_trace = trace + g_trace
            discriminator_prediction = self.discriminator(protected_trace, training=False)
            generator_loss = self.gen_loss(attack_point, discriminator_prediction)
        gradients = tape.gradient(generator_loss, trainable_vars)
        self.gen_optimizer.apply_gradients(zip(gradients, trainable_vars))
        discriminator_loss = self.disc_loss(attack_point, discriminator_prediction)
        generator_loss = generator_loss.numpy()
        discriminator_loss = discriminator_loss.numpy()
        return (generator_loss, discriminator_loss)
    def disc_train_step(self, key, batch):
        (trace, plaintext, attack_point) = batch
        trainable_vars = self.discriminator.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(trainable_vars)
            g_trace = self.generators[key](plaintext, training=False)
            protected_trace = trace + g_trace
            discriminator_prediction = self.discriminator(protected_trace, training=True)
            discriminator_loss = self.disc_loss(attack_point, discriminator_prediction)
        gradients = tape.gradient(discriminator_loss, trainable_vars)
        self.disc_optimizer.apply_gradients(zip(gradients, trainable_vars))
        generator_loss = self.gen_loss(attack_point, discriminator_prediction)
        generator_loss = generator_loss.numpy()
        discriminator_loss = discriminator_loss.numpy()
        return (generator_loss, discriminator_loss)
    def eval_step(self, key, batch):
        (trace, plaintext, attack_point) = batch
        g_trace = self.generators[key](plaintext, training=False)
        protected_trace = trace + g_trace
        discriminator_prediction = self.discriminator(protected_trace, training=False)
        generator_loss = self.gen_loss(attack_point, discriminator_prediction)
        discriminator_loss = self.disc_loss(attack_point, discriminator_prediction)
        generator_loss = generator_loss.numpy()
        discriminator_loss = discriminator_loss.numpy()
        return (generator_loss, discriminator_loss)
    def calculate_saliency(self, key, batch):
        (trace, plaintext, attack_point) = batch
        trace = tf.Variable(trace[0], dtype=float)
        plaintext = np.expand_dims(plaintext[0], axis=0)
        attack_point = np.expand_dims(attack_point[0], axis=0)
        with tf.GradientTape() as tape:
            g_trace = self.generators[key](plaintext, training=False)
            protected_trace = trace + g_trace
            disc_prediction = self.discriminator(protected_trace, training=False)
            prediction_idx = np.argmax(disc_prediction.numpy()[0])
            out = disc_prediction[0][prediction_idx]
        gradients = tape.gradient(out, protected_trace)
        protected_trace = np.squeeze(protected_trace.numpy())
        saliency = np.squeeze(gradients.numpy())
        return (protected_trace, saliency)
    def train(self, dataset,
              num_steps = None,
              gen_pretrain_epochs = None,
              disc_pretrain_epochs = None,
              gen_epochs_per_step = None,
              disc_epochs_per_step = None,
              measure_saliency_period = None):
        if not(isinstance(dataset, datasets.Dataset)):
            raise TypeError('dataset must be instance of class {}'.format(datasets.Dataset))
        if num_steps == None:
            num_steps = 1
            printl('\tUsing default number of steps: {}'.format(num_steps))
        else:
            printl('\tUsing specified number of steps: {}'.format(num_steps))
        if type(num_steps) != int:
            raise TypeError('num_steps must be of type {} but is of type {}'.format(int, type(num_steps)))
        if gen_pretrain_epochs == None:
            gen_pretrain_epochs = 0
            printl('\tUsing default number of generator pretraining epochs: {}'.format(gen_pretrain_epochs))
        else:
            printl('\tUsing specified number of generator pretraining epochs: {}'.format(gen_pretrain_epochs))
        if type(gen_pretrain_epochs) != int:
            raise TypeError('gen_pretrain_epochs must be of type {} but is of type {}'.format(int, type(gen_pretrain_epochs)))
        if disc_pretrain_epochs == None:
            disc_pretrain_epochs = 0
            printl('\tUsing default number of discriminator pretraining epochs: {}'.format(disc_pretrain_epochs))
        else:
            printl('\tUsing specified number of discriminator pretraining epochs: {}'.format(disc_pretrain_epochs))
        if type(disc_pretrain_epochs) != int:
            raise TypeError('disc_pretrain_epochs must be of type {} but is of type {}'.format(int, type(disc_pretrain_epochs)))
        if gen_epochs_per_step == None:
            gen_epochs_per_step = 1
            printl('\tUsing default generator epochs per step: {}'.format(gen_epochs_per_step))
        else:
            printl('\tUsing specified generator epochs per step: {}'.format(gen_epochs_per_step))
        if not(type(gen_epochs_per_step) == int):
            raise TypeError('gen_epochs_per_step must be of type {} but is of type {}'.format(int, type(gen_epochs_per_step)))
        if disc_epochs_per_step == None:
            disc_epochs_per_step = 1
            printl('\tUsing default discriminator epochs per step: {}'.format(disc_epochs_per_step))
        else:
            printl('\tUsing specified discriminator epochs per step: {}'.format(disc_epochs_per_step))
        if not(type(disc_epochs_per_step) == int):
            raise TypeError('disc_epochs_per_step must be of type {} but is of type {}'.format(int, type(disc_epochs_per_step)))
        printl('\tUsing specified saliency measurement period: {}'.format(measure_saliency_period))
        if (measure_saliency_period != None) and not(type(measure_saliency_period) == int):
            raise TypeError('measure_saliency_period must be None or of type {} but is of type {}'.format(int, type(measure_saliency_period)))
        printl()
        
        step = 0
        d_epoch = 0
        d_loss = np.nan
        g_epoch = 0
        g_loss = np.nan
        results = {
            'gen_training_loss': {k: {} for k in self.generators},
            'disc_training_loss': {k: {} for k in self.generators},
            'saliency': {k: {} for k in self.generators}}
        
        if (disc_pretrain_epochs == 0) and (gen_pretrain_epochs == 0):
            # Calculate initial model performance
            printl('Calculating initial model performance:')
            t0 = time.time()
            for key in self.generators:
                printl('\tKey %x:'%(key))
                results['disc_training_loss'][key].update({0: []})
                results['gen_training_loss'][key].update({0: []})
                for batch in dataset.get_batch_iterator(key):
                    g_loss, d_loss = self.eval_step(key, batch)
                    results['disc_training_loss'][key][0].append(d_loss)
                    results['gen_training_loss'][key][0].append(g_loss)
                results['disc_training_loss'][key][0] = np.mean(results['disc_training_loss'][key][0])
                results['gen_training_loss'][key][0] = np.mean(results['gen_training_loss'][key][0])
                printl('\t\tDiscriminator loss: %04e'%(results['disc_training_loss'][key][0]))
                printl('\t\tGenerator loss: %04e'%(results['gen_training_loss'][key][0]))
            time_taken = time.time()-t0
            printl('Done. Time taken: %.04f sec'%(time_taken))
            printl()
        
        if disc_pretrain_epochs != 0:
            printl('Pretraining discriminator:')
            t0 = time.time()
            for epoch in range(-disc_pretrain_epochs-gen_pretrain_epochs+1, -gen_pretrain_epochs+1):
                printl('\tBeginning epoch %d:'%(epoch))
                for key in self.generators:
                    printl('\t\tKey: %x'%(key))
                    results['disc_training_loss'][key].update({epoch: []})
                    results['gen_training_loss'][key].update({epoch: []})
                    for d_batch in dataset.get_batch_iterator(key):
                        g_loss, d_loss = self.disc_train_step(key, d_batch)
                        results['disc_training_loss'][key][epoch].append(d_loss)
                        results['gen_training_loss'][key][epoch].append(g_loss)
                    printl('\t\t\tDiscriminator loss: %04e'%(np.mean(results['disc_training_loss'][key][epoch])))
                    printl('\t\t\tGenerator loss: %04e'%(np.mean(results['gen_training_loss'][key][epoch])))
            time_taken = time.time()-t0
            printl('Done. Time taken: %.04f sec.'%(time_taken))
        
        if gen_pretrain_epochs != 0:
            printl('Pretraining generators:')
            t0 = time.time()
            for epoch in range(-gen_pretrain_epochs+1, 1):
                printl('\tBeginning generator epoch %d:'%(epoch))
                for key in self.generators:
                    printl('\t\tKey: %x'%(key))
                    results['disc_training_loss'][key].update({epoch: []})
                    results['gen_training_loss'][key].update({epoch: []})
                    for g_batch in dataset.get_batch_iterator(key):
                        g_loss, d_loss = self.gen_train_step(key, g_batch)
                        results['disc_training_loss'][key][epoch].append(d_loss)
                        results['gen_training_loss'][key][epoch].append(g_loss)
                    printl('\t\t\tDiscriminator loss: %04e'%(np.mean(results['disc_training_loss'][key][epoch])))
                    printl('\t\t\tGenerator loss: %04e'%(np.mean(results['gen_training_loss'][key][epoch])))
            time_taken = time.time()-t0
            printl('Done. Time taken: %.04f sec.'%(time_taken))
        
        if measure_saliency_period != None:
            t0 = time.time()
            printl('Beginning initial saliency measurement:')
            for key in self.generators:
                printl('\t\tKey: %x'%(key))
                batch = next(dataset.get_batch_iterator(key))
                (protected_trace, saliency) = self.calculate_saliency(key, batch)
                results['saliency'][key].update({0: (protected_trace, saliency)})
            time_taken = time.time()-t0
            printl('Done. Time taken: %.04f sec'%(time_taken))
            printl()
        
        for step in range(1, num_steps+1):
            t0 = time.time()
            printl('Beginning step %d:'%(step))
            d_epoch = 0
            g_epoch = 0
            for epoch in range(1, disc_epochs_per_step+1):
                printl('\tBeginning discriminator epoch %d:'%(epoch))
                for key in self.generators:
                    printl('\t\tKey: %x'%(key))
                    results['disc_training_loss'][key].update({step: []})
                    results['gen_training_loss'][key].update({step: []})
                    for d_batch in dataset.get_batch_iterator(key):
                        g_loss, d_loss = self.disc_train_step(key, d_batch)
                        results['disc_training_loss'][key][step].append(d_loss)
                        results['gen_training_loss'][key][step].append(g_loss)
                    printl('\t\t\tDiscriminator loss: %04e'%(np.mean(results['disc_training_loss'][key][step])))
                    printl('\t\t\tGenerator loss: %04e'%(np.mean(results['gen_training_loss'][key][step])))
            for epoch in range(1, gen_epochs_per_step+1):
                printl('\tBeginning generator epoch %d:'%(epoch))
                for key in self.generators:
                    printl('\t\tKey: %x'%(key))
                    results['disc_training_loss'][key].update({step+.5: []})
                    results['gen_training_loss'][key].update({step+.5: []})
                    for g_batch in dataset.get_batch_iterator(key):
                        g_loss, d_loss = self.gen_train_step(key, g_batch)
                        results['disc_training_loss'][key][step+.5].append(d_loss)
                        results['gen_training_loss'][key][step+.5].append(g_loss)
                    printl('\t\t\tDiscriminator loss: %04e'%(np.mean(results['disc_training_loss'][key][step+.5])))
                    printl('\t\t\tGenerator loss: %04e'%(np.mean(results['gen_training_loss'][key][step+.5])))
            time_taken = time.time()-t0
            printl('Done. Time taken: %.04f sec. ETA: %.04f sec'%(time_taken, time_taken*(num_steps-step)))
            printl()
            
            if (measure_saliency_period != None) and (step%measure_saliency_period == 0):
                t0 = time.time()
                printl('Beginning saliency measurement:')
                for key in self.generators:
                    printl('\t\tKey: %x'%(key))
                    batch = next(dataset.get_batch_iterator(key))
                    (protected_trace, saliency) = self.calculate_saliency(key, batch)
                    results['saliency'][key].update({step: (protected_trace, saliency)})
                time_taken = time.time()-t0
                printl('Done. Time taken: %.04f sec'%(time_taken))
                printl()
        return results
    def save(self, dest, prefix=''):
        printl('Saving models:')
        t0 = time.time()
        for gen_key in self.generators:
            self.generators[gen_key].save(os.path.join(dest, prefix+'gen_%x'%(gen_key)))
        self.discriminator.save(os.path.join(dest, prefix+'disc'))
        time_taken = time.time()-t0
        printl('Done. Time taken: %.04f sec'%(time_taken))
    
def get_mlp_model(key, trace_length,
                  layers=None,
                  hidden_activation=None,
                  output_activation=None,
                  plaintext_encoding=None):
    printl('Generating multilayer perceptron model.')
    if not(type(key) == int):
        raise TypeError('key must be of type {} but is of type {}'.format(int, type(key)))
    printl('\tKey: {}'.format(hex(key)))
    if not(type(trace_length) == int):
        raise TypeError('trace_length must be of type {} but is of type {}'.format(int, type(trace_length)))
    printl('\tTrace length: {}'.format(trace_length))
    if layers == None:
        layers = [64, 64, 64]
        printl('\tUsing default layers: {}'.format(layers))
    else:
        printl('\tUsing specified layers: {}'.format(layers))
    if not(type(layers) == list):
        raise TypeError('layers must be of type {} but is of type {}'.format(list, type(layers)))
    if not(all([type(l) == int for l in layers])):
        raise TypeError('items in layers must be of type {} but layers includes items of type {}'.format(int, ', '.join([type(l) for l in layers if type(l) != int])))
    if hidden_activation == None:
        hidden_activation = 'relu'
        printl('\tUsing default hidden activation: {}'.format(hidden_activation))
    else:
        printl('\tUsing specified hidden activation: {}'.format(hidden_activation))
    if not(type(hidden_activation) == str):
        raise TypeError('hidden_activation must be of type {} but is of type {}'.format(str, type(hidden_activation)))
    if output_activation == None:
        output_activation = 'linear'
        printl('\tUsing default output activation: {}'.format(output_activation))
    else:
        printl('\tUsing specified output activation: {}'.format(output_activation))
    if not(type(output_activation) == str):
        raise TypeError('output_activation must be of type {} but is of type {}'.format(str, type(output_activation)))
    if plaintext_encoding == None:
        plaintext_encoding = 'binary'
        printl('\tUsing default plaintext encoding: {}'.format(plaintext_encoding))
    else:
        printl('\tUsing specified plaintext encoding: {}'.format(plaintext_encoding))
    if not(type(plaintext_encoding) == str):
        raise TypeError('plaintext_encoding must be of type {} but is of type {}'.format(str, type(plaintext_encoding)))
    
    if plaintext_encoding == 'binary':
        input_shape = (8,)
    elif plaintext_encoding == 'scalar':
        input_shape = (1,)
    elif plaintext_encoding == 'onehot':
        input_shape = (256,)
    else:
        raise Exception('Invalid plaintext encoding: \'{}\''.format(plaintext_encoding))
    printl('\tInput shape: {}'.format(input_shape))
    output_shape = (trace_length, 1)
    printl('\tOutput shape: {}'.format(output_shape))

    model_layers = \
        [keras.layers.InputLayer(input_shape)] + \
        [keras.layers.Dense(units, activation=hidden_activation) for units in layers] + \
        [keras.layers.Dense(np.prod(output_shape), activation=output_activation)] + \
        [keras.layers.Reshape(output_shape)]
    model = Generator(model_layers, key)
    return model

def get_zero_model(key, trace_length,
                   plaintext_encoding=None,
                   **kwargs):
    printl('Generating zero-output generator model:')
    if not(type(key) == int):
        raise TypeError('key must be of type {} but is of type {}'.format(int, type(key)))
    printl('\tKey: {}'.format(hex(key)))
    if not(type(trace_length) == int):
        raise TypeError('trace_length must be of type {} but is of type {}'.format(int, type(trace_length)))
    printl('\tTrace length: {}'.format(trace_length))
    if plaintext_encoding == None:
        plaintext_encoding = 'binary'
        printl('\tUsing default plaintext encoding: {}'.format(plaintext_encoding))
    else:
        printl('\tUsing specified plaintext encoding: {}'.format(plaintext_encoding))
    if not(type(plaintext_encoding) == str):
        raise TypeError('plaintext_encoding must be of type {} but is of type {}'.format(str, type(plaintext_encoding)))
    
    if plaintext_encoding == 'binary':
        input_shape = (8,)
    elif plaintext_encoding == 'scalar':
        input_shape = (1,)
    elif plaintext_encoding == 'onehot':
        input_shape = (256,)
    else:
        raise Exception('Invalid plaintext encoding: \'{}\''.format(plaintext_encoding))
    printl('\tInput shape: {}'.format(input_shape))
    output_shape = (trace_length, 1)
    printl('\tOutput shape: {}'.format(output_shape))
    
    model_layers = \
        [keras.layers.InputLayer(input_shape)] + \
        [keras.layers.Dense(np.prod(output_shape), activation='linear', kernel_initializer='zeros', bias_initializer='zeros', trainable=False)] + \
        [keras.layers.Reshape(output_shape)]
    model = Generator(model_layers, key)

def get_random_model(key, trace_length,
                     plaintext_encoding=None,
                     mean=None,
                     var=None,
                     **kwargs):
    printl('Generating random generator model:')
    if not(type(key) == int):
        raise TypeError('key must be of type {} but is of type {}'.format(int, type(key)))
    printl('\tKey: {}'.format(hex(key)))
    if not(type(trace_length) == int):
        raise TypeError('trace_length must be of type {} but is of type {}'.format(int, type(trace_length)))
    printl('\tTrace length: {}'.format(trace_length))
    if plaintext_encoding == None:
        plaintext_encoding = 'binary'
        printl('\tUsing default plaintext encoding: {}'.format(plaintext_encoding))
    else:
        printl('\tUsing specified plaintext encoding: {}'.format(plaintext_encoding))
    if not(type(plaintext_encoding) == str):
        raise TypeError('plaintext_encoding must be of type {} but is of type {}'.format(str, type(plaintext_encoding)))
    if mean == None:
        mean = 0.0
        printl('\tUsing default mean for Gaussian noise: {}'.format(mean))
    else:
        printl('\tUsing specified mean for Gaussian noise: {}'.format(mean))
    if not(type(mean) == float):
        raise TypeError('mean must be of type {} but is of type {}'.format(float, type(mean)))
    if var == None:
        var = 1.0
        printl('\tUsing default variance for Gaussian noise: {}'.format(var))
    else:
        printl('\tUsing specified variance for Gaussian noise: {}'.format(var))
    if not(type(var) == float):
        raise TypeError('var must be of type {} but is of type {}'.format(float, type(var)))
    
    if plaintext_encoding == 'binary':
        input_shape = (8,)
    elif plaintext_encoding == 'scalar':
        input_shape = (1,)
    elif plaintext_encoding == 'onehot':
        input_shape = (256,)
    else:
        raise Exception('Invalid plaintext encoding: \'{}\''.format(plaintext_encoding))
    printl('\tInput shape: {}'.format(input_shape))
    output_shape = (trace_length, 1)
    printl('\tOutput shape: {}'.format(output_shape))
    
    model_layers = \
        [keras.layers.InputLayer(input_shape)] + \
        [keras.layers.Lambda(lambda x: tf.random.normal(output_shape, mean=mean, stddev=var**.5))]
    model = Generator(model_layers, key)

def get_generators(keys, gen_type, trace_length, **kwargs):
    if not(type(keys) == list):
        raise TypeError('keys must be of type {} but is of type {}'.format(list, type(keys)))
    if not(all([type(k) == int for k in keys])):
        raise TypeError('items in keys must be of type {} but keys includes items of type {}'.format(int, ', '.join([type(k) for k in keys if type(k) != int])))
    if not(type(trace_length) == int):
        raise TypeError('trace_length must be of type {} but is of type {}'.format(int, type(trace_length)))
    
    models = {}
    for key in keys:
        if gen_type == 'mlp':
            model = get_mlp_model(key, trace_length, **kwargs)
        elif gen_type == 'zero':
            model = get_zero_model(key, trace_length, **kwargs)
        elif gen_type == 'random':
            model = get_random_model(key, trace_length, **kwargs)
        else:
            raise ValueError('Invalid generator type: {}'.format(gen_type))
        model.summary(print_fn=printl)
        models.update({key: model})
    return models

def normalize_discriminator(discriminator, input_shape):
    trace_inp = keras.layers.Input(shape=input_shape)
    trace = trace_inp
    trace = keras.layers.Lambda(lambda x: x-tf.math.reduce_mean(x))(trace)
    trace = keras.layers.Lambda(lambda x: x/tf.math.reduce_std(x))(trace)
    output = discriminator(trace)
    model = keras.Model(inputs=trace_inp, outputs=output, name='Discriminator')
    return model

def get_discriminator(disc_type, trace_length, **kwargs):
    if not(type(disc_type) == str):
        raise TypeError('disc_type must be of type {} but is of type {}'.format(str, type(disc_type)))
    if not(type(trace_length) == int):
        raise TypeError('trace_length must be of type {} but is of type {}'.format(int, type(trace_length)))
    
    if disc_type == 'google_resnet1d':
        model = google_resnet1d.Resnet1D((trace_length, 1), printl, **kwargs)
    elif disc_type == 'google_resnet1d_pretrained':
        model = google_resnet1d.PretrainedResnet1D(printl, **kwargs)
    else:
        raise ValueError('Invalid discriminator type: {}'.format(disc_type))
    model.summary(print_fn=printl)
    discriminator = normalize_discriminator(model, (trace_length, 1))
    return discriminator

def get_gan(generators, discriminator, **kwargs):
    if not(type(generators) == dict):
        raise TypeError('generators must be of type {} but is of type {}'.format(dict, type(generators)))
    if not(all([isinstance(generators[k], Generator) for k in generators])):
        raise TypeError('all elements in generators must be instance of class {}'.format(Generator))
    if not(isinstance(discriminator, keras.Model)):
        raise TypeError('discriminator must be instance of class {}'.format(Discriminator))
    
    gan = GAN(generators, discriminator, **kwargs)
    return gan