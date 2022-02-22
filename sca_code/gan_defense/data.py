import os
import numpy as np

def load_shard(key, attack_point, byte, train):
    path = os.path.join(os.getcwd(), 'datasets', 'tinyaes', 'train' if train else 'test')
    file = None
    for f in os.listdir(path):
        if f == '%032x.npz'%(key):
            file = os.path.join(path, f)
    if file == None:
        raise ValueError('File \'%032x.npz\' not found in folder %s'%(key, path))
    data = np.load(file)
    traces = np.squeeze(data['traces'][:20000])
    attack_point = data[attack_point][byte]
    plaintext = data['pts'][byte]
    return {'traces': traces, 'plaintext': plaintext, 'attack_point': attack_point}

def load_dataset(keys, train=True, attack_point='sub_bytes_in', byte=0):
    shards = {}
    for key in keys:
        shard = load_shard(key, attack_point, byte, train)
        shards.update({key: shard})
    return shards