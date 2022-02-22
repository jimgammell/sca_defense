import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras

import utils
import google_resnet1d

def printl(s=''):
    utils.printl('(models): ' + s)

class Generator(keras.Sequential):
    def __init__(self, model_layers, key):
        super(Generator, self).__init__(model_layers, name='Generator_%s'%(hex(key)))
        self.key = key

class GAN:
    def __init__(self, 
                 generators,
                 discriminator,
                 gen_optimizer='SGD',
                 gen_optimizer_kwargs={},
                 gen_loss='CategoricalCrossentropy',
                 gen_loss_kwargs={},
                 disc_optimizer='Adam',
                 disc_optimizer_kwargs={'lr': 0.001},
                 disc_loss='CategoricalCrossentropy',
                 disc_loss_kwargs={}):
        valid_optimizers = {
            'SGD': keras.optimizers.SGD,
            'RMSprop': keras.optimizers.RMSprop,
            'Adam': keras.optimizers.Adam}
        valid_losses = {
            'CategoricalCrossentropy': keras.losses.CategoricalCrossentropy}
        if gen_optimizer in valid_optimizers:
            gen_optimizer = valid_optimizers[gen_optimizer]
        else:
            raise Exception('Invalid gen_optimizer: \'{}\''.format(gen_optimizer))
        if disc_optimizer in valid_optimizers:
            disc_optimizer = valid_optimizers[disc_optimizer]
        else:
            raise Exception('Invalid disc_optimizer: \'{}\''.format(disc_optimizer))
        if gen_loss in valid_losses:
            gen_loss = valid_losses[gen_loss]
        else:
            raise Exception('Invalid gen_loss: \'{}\''.format(gen_loss))
        if disc_loss in valid_losses:
            disc_loss = valid_losses[disc_loss]
        else:
            raise Exception('Invalid disc_loss: \'{}\''.format(disc_loss))
            
        self.generators = generators
        self.gen_optimizer = gen_optimizer(**gen_optimizer_kwargs)
        self.gen_loss = lambda label, pred: -gen_loss(**gen_loss_kwargs)(label, pred)
        self.discriminator = discriminator
        self.disc_optimizer = disc_optimizer(**disc_optimizer_kwargs)
        self.disc_loss = disc_loss(**disc_loss_kwargs)
        self.results = {
            'disc_training_loss': {'epoch': [], 'loss': []},
            'gen_training_loss': {'epoch': [], 'loss': {key: [] for key in self.generators}},
            'disc_test_loss': {'epoch': [], 'loss': []},
            'gen_test_loss': {'epoch': [], 'loss': {key: [] for key in self.generators}},
            'disc_training_accuracy': {'epoch': [], 'accuracy': []},
            'gen_training_accuracy': {'epoch': [], 'accuracy': {key: [] for key in self.generators}},
            'disc_test_accuracy': {'epoch': [], 'accuracy': []},
            'gen_test_accuracy': {'epoch': [], 'accuracy': {key: [] for key in self.generators}}}
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
        return generator_loss
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
        return discriminator_loss
    def gen_eval_step(self, key, batch):
        pass
    def disc_eval_step(self, key, batch):
        pass
    def train(self, dataset,
              num_steps=100,
              gen_epochs_per_step=1,
              disc_epochs_per_step=1):
        step = 0
        d_epoch = 0
        d_loss = np.nan
        g_epoch = 0
        g_loss = np.nan
        def tqdm_update(t):
            t.set_description('Step: %d, Disc loss: %.05f, Gen loss: %.05f'%(step, d_loss, g_loss))
        for step in range(1, num_steps+1):
            d_epoch = 0
            g_epoch = 0
            for d_epoch in range(1, disc_epochs_per_step+1):
                d_losses = []
                with tqdm.tqdm(self.generators, position=0, leave=True, ncols=100) as t:
                    for d_key in t:
                        d_losses_k = []
                        for d_batch in dataset.get_batch_iterator(d_key):
                            d_loss = self.disc_train_step(d_key, d_batch)
                            d_losses_k.append(d_loss)
                            tqdm_update(t)
                        d_losses.append(np.mean(d_losses_k))
                    self.results['disc_training_loss']['epoch'].append(step)
                    self.results['disc_training_loss']['loss'].append(d_losses)
            for g_epoch in range(1, gen_epochs_per_step+1):
                with tqdm.tqdm(self.generators, position=0, leave=True, ncols=100) as t:
                    for g_key in t:
                        g_losses = []
                        for g_batch in dataset.get_batch_iterator(g_key):
                            g_loss = self.gen_train_step(g_key, g_batch)
                            g_losses.append(g_loss)
                            tqdm_update(t)
                        self.results['gen_training_loss']['loss'][g_key].append(g_losses)
                    self.results['gen_training_loss']['epoch'].append(step)
    def calculate_saliency(self, key, trace):
        pass

def get_mlp_model(key, trace_length, **kwargs):
    printl('Generating multilayer perceptron model.')
    printl('\tKey: %s'%(hex(key)))
    printl('\tTrace length: %d'%(trace_length))
    layers = [64, 64, 64] if not('layers' in kwargs) else kwargs['layers']
    hidden_activation = 'relu' if not('hidden_activation' in kwargs) else kwargs['hidden_activation']
    output_activation = 'linear' if not('output_activation' in kwargs) else kwargs['output_activation']
    plaintext_encoding = 'binary' if not('plaintext_encoding' in kwargs) else kwargs['plaintext_encoding']
    if plaintext_encoding == 'binary':
        input_shape = (8,)
    elif plaintext_encoding == 'scalar':
        input_shape = (1,)
    elif plaintext_encoding == 'onehot':
        input_shape = (256,)
    else:
        raise Exception('Invalid plaintext encoding: \'{}\''.format(plaintext_encoding))
    output_shape = (trace_length, 1)
    printl('\tHidden activation: {}'.format(hidden_activation))
    printl('\tOutput activation: {}'.format(output_activation))
    printl('\tPlaintext encoding format: {}'.format(plaintext_encoding))

    model_layers = \
        [keras.layers.InputLayer(input_shape)] + \
        [keras.layers.Dense(units, activation=hidden_activation) for units in layers] + \
        [keras.layers.Dense(np.prod(output_shape), activation=output_activation)] + \
        [keras.layers.Reshape(output_shape)]
    model = Generator(model_layers, key)
    printl('\tDone.')
    return model

def get_generators(keys, gen_type, trace_length, **kwargs):
    models = {}
    for key in keys:
        if gen_type == 'mlp':
            model = get_mlp_model(key, trace_length, **kwargs)
            model.summary(print_fn=printl)
            models.update({key: model})
        else:
            raise Exception('Invalid generator type: {}'.format(gen_type))
    return models

def get_discriminator(disc_type, trace_length, **kwargs):
    if disc_type == 'google_resnet1d':
        model = google_resnet1d.Resnet1D((trace_length, 1), **kwargs)
        model.summary(print_fn=printl)
    else:
        raise Exception('Invalid discriminator type: {}'.format(disc_type))
    return model

def get_gan(generators, discriminator, **kwargs):
    gan = GAN(generators, discriminator, **kwargs)
    return gan