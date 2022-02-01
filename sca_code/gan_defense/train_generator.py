# To do:
#   Try using gradient of discriminator in objective of generator
#   Try minimizing inf-norm of discriminator prediction, rather than magnitude of correct guess

import os
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import time
import datetime
from keras.utils import to_categorical

N_TRACES = 16 # Number of traces per AES key (training and test datasets)
DATA_PATH = os.path.join(os.getcwd(), 'datasets', 'tinyaes') # Where the dataset to train on is stored
OUTPUT_PATH = os.path.join(os.getcwd(), 'results') # Folder in which results will be saved
dt = datetime.datetime.now()
OUTPUT_PATH = os.path.join(OUTPUT_PATH,
    '%d-%d-%d_%d-%d-%d'%(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second))
assert not(os.path.exists(OUTPUT_PATH))
os.makedirs(OUTPUT_PATH) # Folder within results folder in which output of this trial will be stored
N_EPOCHS = 100 # Number of epochs to train for during each phase of trial (initial discriminator training, generator training, discriminator training on generator outputs
BATCH_SIZE = 32 # Number of trace/AP pairs per batch during training

# Function to be used in place of print -- prints both to the terminal and to a log file
def printl(s):
    print(s)
    with open(os.path.join(OUTPUT_PATH, 'log.txt'), 'a') as F:
        F.write('%s\n'%(s))

# Function that converts attack point value to binary vector, i.e. if attack point is 59=0b00111011, resulting vector will be (0, 0, 1, 1, 1, 0, 1, 1)
def int_to_ohbinary(y_int):
    y = [np.zeros((1, 8)) for _ in range(len(y_int))]
    for (idx, yy) in enumerate(y_int):
        bit = -1
        base = 256
        while base > 1:
            base /= 2
            bit += 1
            if yy >= base:
                yy -= base
                y[idx][0, bit] = 1.0
            else:
                y[idx][0, bit] = 0.0
        assert yy == 0
    y = [tf.convert_to_tensor(yy, dtype='float32') for yy in y]
    y = tf.stack(y, axis=0)
    return y

# Function to train one part of the cumulative model
def train_model(mdl, disc, gen, disc_trainable, gen_trainable, time_input):
    assert disc_trainable != gen_trainable
    
    def gen_cost(y_true, y_pred):
        return tf.norm(y_pred, ord=np.inf)
    def disc_cost(y_true, y_pred):
        return CategoricalCrossentropy()(y_true, y_pred)
    
    # Train either the generator or discriminator for 100 epochs or until validation loss does not improve for 5 epochs -- whichever comes first
    #callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
    #                                            patience=100,
    #                                            restore_best_weights=False)
    disc.trainable = disc_trainable
    gen.trainable = gen_trainable
    printl('Disc trainable: %s. Gen trainable: %s.'%(disc_trainable, gen_trainable))
    if disc_trainable:
        loss = disc_cost
        optimizer = Adam(lr=.001)
    else:  
        loss = gen_cost
        optimizer = Adam(lr=.0001)
    mdl.compile(loss=loss,
                optimizer=optimizer,
                metrics=['categorical_accuracy', gen_cost, disc_cost])
    if time_input:
        print('Training with provided time vector.')
        hist = mdl.fit([ap_train, traces_train, times], targets_train,
                validation_data=([ap_test, traces_test, times], targets_test),
                shuffle=True, epochs=(5*N_EPOCHS if gen_trainable else N_EPOCHS),
                       batch_size=BATCH_SIZE, verbose=2)
                #callbacks=[callback], verbose=2)
    else:
        print('Training with no provided time vector.')
        hist = mdl.fit([ap_train, traces_train], targets_train,
                validation_data=([ap_test, traces_test], targets_test),
                shuffle=True, epochs=N_EPOCHS, 
                batch_size=BATCH_SIZE, verbose=2)
                #callbacks=[callback], verbose=2)
    
    # Plot the training/validation loss/accuracy during training
    t_fig = plt.figure()
    ax = plt.gca()
    ax.plot(hist.history['loss'], '--', color='blue', label='Training loss')
    ax.plot(hist.history['val_loss'], '-', color='blue', label='Validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    tax = ax.twinx()
    tax.plot(hist.history['categorical_accuracy'], '--', color='red', label='Training accuracy')
    tax.plot(hist.history['val_categorical_accuracy'], '-', color='red', label='Validation accuracy')
    tax.plot(np.ones(len(hist.history['categorical_accuracy']))/256, '--', color='black', label='Baseline: 1/256')
    tax.legend()
    tax.set_ylabel('Accuracy (proportion)')
    tax.set_ylim([0, 1])
    
    acc1 = mdl.evaluate([ap_test, traces_test], targets_test, batch_size=1)
    acc16 = mdl.evaluate([ap_test, traces_test], targets_test, batch_size=16)
    acc32 = mdl.evaluate([ap_test, traces_test], targets_test, batch_size=32)
    acc64 = mdl.evaluate([ap_test, traces_test], targets_test, batch_size=64)
    acc128 = mdl.evaluate([ap_test, traces_test], targets_test, batch_size=128)
    printl('Keras-computed evaluation with batch size 1: {}.'.format(acc1))
    printl('Keras-computed evaluation with batch size 16: {}.'.format(acc16))
    printl('Keras-computed evaluation with batch size 32: {}.'.format(acc32))
    printl('Keras-computed evaluation with batch size 64: {}.'.format(acc64))
    printl('Keras-computed evaluation with batch size 128: {}.'.format(acc128))
    
    # Print confusion matrix of discriminator on generator-produced traces
    confusion_matrix = np.zeros((256, 256))
    n_correct = 0
    if time_input:
        predictions = mdl.predict([ap_test, traces_test, times_e], batch_size=BATCH_SIZE)
    else:
        predictions = mdl.predict([ap_test, traces_test], batch_size=BATCH_SIZE)
    printl('Predictions shape: {}'.format(predictions.shape))
    for (prediction, key) in zip(predictions, keys_test):
        c_preds = np.argmax(prediction)
        if c_preds==key:
            n_correct += 1
        confusion_matrix[key, c_preds] += 1
    n_correct /= len(traces_test)
    printl('Done evaluating network. Proportion correct: %f.'%(n_correct))
    cm_fig = plt.figure()
    ax = plt.gca()
    img = ax.imshow(confusion_matrix, cmap='plasma', interpolation='nearest', aspect='equal')
    ax.set_xlabel('Correct attack point')
    ax.set_ylabel('Predicted attack point')
    plt.colorbar(img, ax=ax)
    
    return {'history': hist.history,
            'confusion matrix': confusion_matrix,
            'training figure': t_fig,
            'confusion matrix figure': cm_fig}

printl('Results will be stored in %s.'%(OUTPUT_PATH))

# Initializing training dataset
printl('Generating training data...')
t0 = time.time()
X, Y, targets = [], [], []
files = [f for f in os.listdir(os.path.join(DATA_PATH, 'train'))
         if os.path.isfile(os.path.join(DATA_PATH, 'train', f))]
for (idx, f) in enumerate(files):
    shard = np.load(os.path.join(DATA_PATH, 'train', f))
    
    x = shard['traces'][:N_TRACES, :20000, :]
    x = tf.convert_to_tensor(x, dtype='float32')
    
    y_int = shard['sub_bytes_in'][0][:N_TRACES]
    y = int_to_ohbinary(y_int)
    
    target = shard['sub_bytes_in'][0][:N_TRACES]
    target = to_categorical(target, 256)
    target = tf.convert_to_tensor(target)
    
    X.append(x)
    Y.append(y)
    targets.append(target)
traces_train = tf.concat(X, axis=0)
traces_train = tf.squeeze(traces_train)
ap_train = tf.concat(Y, axis=0)
ap_train = tf.squeeze(ap_train)
targets_train = tf.concat(targets, axis=0)
targets_train = tf.squeeze(targets_train)
times = [t for t in range(20000)]
times = tf.constant(times, dtype='float32')
times = tf.expand_dims(times, axis=0)
times = [times for _ in range(256*N_TRACES)]
times = tf.concat(times, axis=0)
times_e = [t for t in range(20000)]
times_e = tf.constant(times_e, dtype='float32')
times_e = tf.expand_dims(times_e, axis=0)

printl('\tDone. Time taken: %f seconds.'%(time.time()-t0))

# Initializing test dataset
printl('Generating test data...')
t0 = time.time()
X, Y, targets, keys = [], [], [], []
files = [f for f in os.listdir(os.path.join(DATA_PATH, 'test'))
         if os.path.isfile(os.path.join(DATA_PATH, 'test', f))]
for (idx, f) in enumerate(files):
    shard = np.load(os.path.join(DATA_PATH, 'test', f))
    
    x = shard['traces'][:N_TRACES, :20000, :]
    x = tf.convert_to_tensor(x, dtype='float32')
    
    y_int = shard['sub_bytes_in'][0][:N_TRACES]
    y = int_to_ohbinary(y_int)
    
    target = shard['sub_bytes_in'][0][:N_TRACES]
    target = to_categorical(target, 256)
    target = tf.convert_to_tensor(target)
    
    key = shard['sub_bytes_in'][0][:N_TRACES]
    
    X.append(x)
    Y.append(y)
    targets.append(target)
    keys.append(np.array(key))
traces_test = tf.concat(X, axis=0)
traces_test = tf.squeeze(traces_test)
ap_test = tf.concat(Y, axis=0)
ap_test = tf.squeeze(ap_test)
targets_test = tf.concat(targets, axis=0)
targets_test = tf.squeeze(targets_test)
keys_test = np.concatenate(keys)
printl('\tDone. Time taken: %f seconds.'%(time.time()-t0))

printl('Keys shape: {}.'.format(keys_test.shape))
printl('Targets shape: {}.'.format(targets_test.shape))
n_matching = 0
for (key, target) in zip(keys_test, targets_test):
    if key == np.argmax(target):
        n_matching += 1
printl('Proportion of targets/keys which match: {}.'.format(n_matching/len(keys_test)))

from generator import IdentityGenerator, LinearGenerator, Mlp1Generator, Mlp3Generator, CnnTransposeGenerator, FourierGenerator
from discriminator import NormalizedDiscriminator
from models import cumulative_model
from keras.losses import CategoricalCrossentropy
from keras.utils import plot_model
from matplotlib import pyplot as plt
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
from scaaml.aes import ap_preds_to_key_preds
from scaaml.utils import from_categorical
import pickle
from scipy.stats import linregress
from keras.utils.layer_utils import count_params
from keras.optimizers import Adam

for gen_fn in [Mlp3Generator]:#[IdentityGenerator, LinearGenerator, Mlp1Generator, Mlp3Generator, CnnTransposeGenerator, FourierGenerator]:
    printl('Beginning trial with %s generator.'%(gen_fn.__name__))
    t0 = time.time()
    
    # Initializing discriminator model
    printl('\tCreating discriminator...')
    t1 = time.time()
    disc = NormalizedDiscriminator(0, 'sub_bytes_in')
    disc.summary(print_fn=printl)
    plot_model(disc, show_shapes=True, to_file=os.path.join(OUTPUT_PATH, 'discriminator_model_%s.png'%(gen_fn.__name__)))
    printl('\t\tDone. Time taken: %f seconds.'%(time.time()-t1))
    
    # Initializing generator model
    printl('\tCreating generator...')
    t1 = time.time()
    gen = gen_fn()
    gen.summary(print_fn=printl)
    plot_model(gen, show_shapes=True, to_file=os.path.join(OUTPUT_PATH, 'generator_model_%s.png'%(gen_fn.__name__)))
    printl('\t\tDone. Time taken: %f seconds.'%(time.time()-t1))
    
    # Initializing full model containing discriminator and generator
    printl('\tCreating cumulative model...')
    t1 = time.time()
    time_input = gen_fn==FourierGenerator
    mdl = cumulative_model(disc, gen, time_input=time_input)
    mdl.summary(print_fn=printl)
    def negative_ccc(y_true, y_pred):
        return -CategoricalCrossentropy()(y_true, y_pred)
    plot_model(mdl, show_shapes=True, to_file=os.path.join(OUTPUT_PATH, 'cumulative_model_%s.png'%(gen_fn.__name__)))
    printl('\t\tDone. Time taken: %f seconds.'%(time.time()-t1))
    
    # Training discriminator with added trace normalization and random generator initialization
    printl('\tTraining modified discriminator...')
    t1 = time.time()
    discini_data = train_model(mdl, disc, gen, True, False, time_input)
    discini_data['training figure'].savefig(os.path.join(OUTPUT_PATH, 'training_discini_%s.png'%(gen_fn.__name__)))
    del discini_data['training figure']
    discini_data['confusion matrix figure'].savefig(os.path.join(OUTPUT_PATH, 'performance_discini_%s.png'%(gen_fn.__name__)))
    del discini_data['confusion matrix figure']
    disc.save(os.path.join(OUTPUT_PATH, 'initial_discriminator_%s'%(gen_fn.__name__)))
    printl('\t\tDone. Time taken: %f seconds.'%(time.time()-t1))
    
    # Training generator with negative discriminator loss
    printl('\tTraining generator...')
    t1 = time.time()
    gen_data = train_model(mdl, disc, gen, False, True, time_input)
    gen_data['training figure'].savefig(os.path.join(OUTPUT_PATH, 'gen_training_%s.png'%(gen_fn.__name__)))
    del gen_data['training figure']
    gen_data['confusion matrix figure'].savefig(os.path.join(OUTPUT_PATH, 'gen_performance_%s.png'%(gen_fn.__name__)))
    del gen_data['confusion matrix figure']
    gen.save(os.path.join(OUTPUT_PATH, 'generator_%s'%(gen_fn.__name__)))
    printl('\t\tDone. Time taken: %f seconds.'%(time.time()-t1))
    
    # Recording a few examples of power traces before/after being passed through generator
    printl('\tPlotting some power traces...')
    t1 = time.time()
    traces = {}
    for i in [0, 500, 1000, 1500, 2000]:
        traces[i] = {}
        x = tf.expand_dims(traces_test[i], axis=0)
        y = tf.expand_dims(ap_test[i], axis=0)
        if time_input:
            visible_trace = gen.predict([y, x, times_e])
        else:
            visible_trace = gen.predict([y, x])
        x = traces_test[i]
        x -= np.mean(x)
        x /= np.std(x)
        x = np.squeeze(x)
        visible_trace -= np.mean(visible_trace)
        visible_trace /= np.std(visible_trace)
        visible_trace = np.squeeze(visible_trace)
        traces[i]['original trace'] = x
        traces[i]['visible trace'] = visible_trace
        (fig, ax) = plt.subplots(1, 3, sharey=True, figsize=(15, 5))
        ax[0].plot(x, '.', color='blue', markersize=.5)
        ax[0].set_title('Original trace')
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Magnitude')
        ax[1].plot(visible_trace, '.', color='blue', markersize=.5)
        ax[1].set_title('Visible trace')
        ax[1].set_xlabel('Time')
        ax[2].plot(visible_trace-x, '.', color='blue', markersize=.5)
        ax[2].set_xlabel('Time')
        ax[2].set_title('Difference')
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_PATH, 'traces_%s_%d.png'%(gen_fn.__name__, i)))
    printl('\t\tDone. Time taken: %f seconds.'%(time.time()-t1))
    
    # Training the discriminator again on the traces produced by the generator
    printl('\tRetraining discriminator on protected trace...')
    t1 = time.time()
    disc_data = train_model(mdl, disc, gen, True, False, time_input)
    disc_data['training figure'].savefig(os.path.join(OUTPUT_PATH, 'disc_training_%s.png'%(gen_fn.__name__)))
    del disc_data['training figure']
    disc_data['confusion matrix figure'].savefig(os.path.join(OUTPUT_PATH, 'disc_performance_%s.png'%(gen_fn.__name__)))
    del disc_data['confusion matrix figure']
    disc.save(os.path.join(OUTPUT_PATH, 'discriminator_%s'%(gen_fn.__name__)))
    printl('\t\tDone. Time taken: %f seconds.'%(time.time()-t1))
    
    # Saving results
    results = {
        'initial discriminator data': discini_data,
        'generator data': gen_data,
        'discriminator data': disc_data,
        'traces': traces}
    with open(os.path.join(OUTPUT_PATH, 'results_%s.pickle'%(gen_fn.__name__)), 'wb') as F:
        pickle.dump(results, F)
    plt.close('all')
    
    printl('\tDone with trial. Time taken: %f seconds.'%(time.time()-t0))
    