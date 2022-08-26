import time
import random
import os
import pkgutil
import importlib
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
        args = [('('+prefix+')').ljust(16)] + list(args)
        print(*args, **kwargs)
        if log_file != None:
            print(*args, file=log_file, **kwargs)
        else:
            print_buffer.append((args, kwargs))
    return print_to_log
        
def specify_log_path(path, save_buffer=False):
    global log_file
    global print_buffer
    if log_file != None:
        log_file.close()
    if path != None:
        log_file = open(path, 'a')
        if save_buffer:
            for (args, kwargs) in print_buffer:
                print(*args, file=log_file, **kwargs)
    else:
        log_file = None
    print_buffer = []

def list_module_attributes(module):
    return [attr for attr in dir(module) if not(attr[:2]=='__' and attr[-2:]=='__')]
        
def get_package_module_names(package):
    package_name = os.path.dirname(package.__file__).split('/')[-1]
    module_names = [name for _, name, _ in pkgutil.iter_modules([package_name])]
    return module_names, package_name
        
def get_package_modules(package):
    module_names, package_name = get_package_module_names(package)
    modules = [importlib.import_module('.'+module_name, package=package_name)
               for module_name in module_names]
    return modules