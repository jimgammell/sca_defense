import time
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

log_file = None
print_buffer = []
def get_print_to_log(prefix):
    def print_to_log(*args, **kwargs):
        global log_file
        global print_buffer
        args = [('('+prefix+')').ljust(20)] + args
        print(*args, **kwargs)
        if log_file != None:
            print(*args, file=log_file, **kwargs)
        else:
            print_buffer.append((args, kwargs))
    return print_to_log_file
        
def specify_log_path(path, save_buffer=False):
    global log_file
    global print_buffer
    if path != None:
        log_file = open(path, 'a')
        if save_buffer:
            for (args, kwargs) in print_buffer:
                print(*args, file=log_file, **kwargs)
        save_buffer = []            
    else:
        log_file = None