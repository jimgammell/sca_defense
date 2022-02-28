import os
import datetime
import numpy as np
import tensorflow as tf

log_file = None
seed = None

def init_printl(path):
    global log_file
    if not os.path.isdir(path):
        raise Exception('Directory not found at path {}.'.format(path))
    log_file = open(os.path.join(path, 'log.txt'), 'a')

def printl(s='', to_terminal=True, to_file=True):
    if to_file:
        global log_file
        if log_file == None:
            raise Exception('Log file not found. Ensure function \'init_printl\' has been run before function \'printl\'.')
        print(s, file=log_file)
    if to_terminal:
        print(s)

def set_random_seed(_seed):
    global seed
    seed = _seed
    tf.random.set_seed(seed)
    np.random.seed(seed)

def get_random_seed():
    global seed
    return seed