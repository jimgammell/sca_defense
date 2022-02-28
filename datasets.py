import os
import numpy as np
import tensorflow as tf

import utils

def printl(s=''):
    utils.printl('(datasets): ' + s)

class Dataset:
    def __init__(self, data, gen_batch_size=16, disc_batch_size_per_key=1):
        self.valid_keys = data.keys()
        self.datasets = {}
        self.gen_batch_size = gen_batch_size
        self.disc_batch_size = disc_batch_size_per_key
        if self.disc_batch_size > self.gen_batch_size:
            raise Exception('Invalid discriminator/generator batch sizes: {}/{}. Discriminator batch size must not exceed generator batch size.'.format(gen_batch_size, disc_batch_size_per_key))
        for key in self.valid_keys:
            self.datasets.update({key: tf.data.Dataset.from_tensor_slices((data[key]['traces'], data[key]['plaintexts'], data[key]['attack_points']))})
            self.datasets[key] = self.datasets[key].batch(gen_batch_size)
            self.datasets[key] = self.datasets[key].shuffle(self.datasets[key].cardinality(), seed=utils.get_random_seed(), reshuffle_each_iteration=True)
    def get_valid_keys(self):
        return self.valid_keys
    def get_batch_iterator(self, key):
        return self.datasets[key].as_numpy_iterator()

def binary(values, depth):
    binary_array = []
    for value in values:
        binary_value = np.zeros((depth,))
        d = depth
        d -= 1
        while d >= 0:
            if value >= 2**d:
                value -= 2**d
                binary_value[d] = 1
            d -= 1
        binary_array.append(binary_value)
    binary_array = np.stack(binary_array, axis=0)
    return binary_array
    
def get_dataset(name, plaintext_encoding='binary', num_keys=16, trace_length=20000, attack_point='sub_bytes_in', byte=0):
    data = {}
    printl('Loading dataset: \'{}\''.format(name))
    printl('\tNumber of keys: {}'.format(num_keys))
    printl('\tAttack point: \'{}\''.format(attack_point))
    printl('\tByte: {}'.format(byte))
    printl('\tTrace length: {}'.format(trace_length))
    if name == 'google_scaaml':
        valid_attack_points = ['keys', 'sub_bytes_in', 'sub_bytes_out']
        if not(attack_point in valid_attack_points):
            raise Exception('Attack point \'{}\' is not valid for dataset \'{}\'.'.format(attack_point, name))
        path = os.path.join(os.getcwd(), 'datasets', 'tinyaes', 'train')
        printl('\tLoading data from \'{}\''.format(path))
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        keys = files[:num_keys]
        printl('\tKeys to be used:')
        for key in keys:
            key_int = int(key[:-4], base=16)
            printl('\t\t%s'%(hex(key_int)))
            data.update({key_int: {}})
            shard = np.load(os.path.join(path, key))
            traces = np.array(shard['traces'][:, :trace_length, :])
            plaintexts = np.array(shard['pts'][byte])
            if plaintext_encoding == 'binary':
                plaintexts = binary(plaintexts, 8)
            elif plaintext_encoding == 'onehot':
                plaintexts = np.array(tf.one_hot(plaintexts, 256))
            elif plaintext_encoding == 'scalar':
                pass
            else:
                raise Exception('Invalid plaintext encoding: {}'.format(plaintext_encoding))
            attack_points = np.array(shard[attack_point][byte])
            attack_points = np.array(tf.one_hot(attack_points, 256))
            data[key_int].update({'traces': traces})
            data[key_int].update({'plaintexts': plaintexts})
            data[key_int].update({'attack_points': attack_points})
        printl('\tConstructing dataset.')
        dataset = Dataset(data)
        printl('\t\tDone.')
    else:
        raise Exception('Dataset name \'{}\' is undefined.')
    return dataset