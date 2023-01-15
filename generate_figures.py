import os
import pickle
import re
import numpy as np
from matplotlib import pyplot as plt

def load_traces(base_dir, keys=None, epochs=None, phases=None):
    def load_results_file(f):
        with open(os.path.join(base_dir, f), 'rb') as F:
            results = pickle.load(F)
        return results
    files = os.path.listdir(os.path.join(base_dir))
    results_files = [f for f in files if re.match('[a-z]+_res_[0-9]+.pickle', f) is not None]
    train_results_files = [f for f in results_files if 'train' in f]
    test_results_files = [f for f in results_files if 'test' in f]
    if epochs is None:
        epochs = [int(s.split('.')[0].split('_')[-1]) for s in train_results_files]
        assert epochs == [int(s.split('.')[0].split('_')[-1]) for s in test_results_files]
    if keys is None:
        keys = [k for k in load_results_file(train_results_files[0]).keys()]
        for f in train_results_files[1:]+test_results_files:
            assert keys == [k for k in load_results_file(f).keys()]
    traces = {
        'epochs': epochs,
        **{'train_'+str(key): [load_results_file(f)[key] for f in train_results_files] for key in keys},
        **{'test_'+str(key): [load_results_file(f)[key] for f in test_results_files] for key in keys}
    }
    return traces