import os
import datetime
import random
import numpy as np
import tensorflow as tf

log_file = None
seed = None

def init_printl(path: str):
    global log_file
    if not os.path.isdir(path):
        raise Exception('Directory not found at path {}.'.format(path))
    log_file = open(os.path.join(path, 'log.txt'), 'a')

def printl(s: str = '', to_terminal: bool = True, to_file: bool = True):
    if to_file:
        global log_file
        if log_file == None:
            raise Exception('Log file not found. Ensure function \'init_printl\' has been run before function \'printl\'.')
        print(s, file=log_file)
    if to_terminal:
        print(s)

def set_random_seed(_seed: int):
    if not(type(_seed) == int):
        raise TypeError('Parameter _seed must be of type {} but is of type {}'.format(int, type(_seed)))
    if not(0 <= _seed <= 0xFFFFFFFF):
        raise ValueError('Parameter _seed must be between {} and {}, but has value {}'.format(hex(0), hex(0xFFFFFFFF), hex(_seed)))
    global seed
    seed = _seed
    random.seed(_seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_random_seed() -> int:
    global seed
    if seed == None:
        raise Exception('Seed must be specified through set_random_seed before get_random_seed can be called')
    return seed

def get_pad_width() -> int:
    return 16