import os
import random
import numpy as np
import tensorflow as tf

import utils

def printl(s=''):
    utils.printl('(datasets):'.ljust(utils.get_pad_width()) + s)

class Dataset:
    def __init__(self, data: dict,
                 gen_batch_size: int
                ):
        if not(type(data) == dict):
            raise TypeError('data must be of type {} but has type {}'.format(dict, type(data)))
        if not(type(gen_batch_size) == int):
            raise TypeError('gen_batch_size must be of type {} but has type {}'.format(int, type(gen_batch_size)))
        
        printl('Generating dataset.')
        self.valid_keys = list(data.keys())
        printl('\tKeys in dataset: {}'.format(', '.join([hex(k) for k in self.valid_keys])))
        self.datasets = {}
        self.gen_batch_size = gen_batch_size
        printl('\tGenerator batch size: {}'.format(self.gen_batch_size))
        num_samples = len(data[self.valid_keys[0]]['traces'])
        printl('\tNumber of samples per key: {}'.format(num_samples))
        num_batches = int((num_samples/self.gen_batch_size) + ((num_samples%self.gen_batch_size) != 0))
        printl('\tNumber of batches per key: {}'.format(num_batches))
        for key in self.valid_keys:
            if len(data[key]['traces']) != num_samples:
                raise Exception('Length {} of traces entry for key {} does not match number of samples {}'.format(len(data[key]['traces']), key, num_samples))
            if len(data[key]['plaintexts']) != num_samples:
                raise Exception('Length {} of plaintexts entry for key {} does not match number of samples {}'.format(len(data[key]['plaintexts']), key, num_samples))
            if len(data[key]['attack_points']) != num_samples:
                raise Exception('Length {} of attack_points entry for key {} does not match number of samples {}'.format(len(data[key]['attack_points']), key, num_samples))
            self.datasets.update({key: tf.data.Dataset.from_tensor_slices((data[key]['traces'], data[key]['plaintexts'], data[key]['attack_points']))})
            if self.datasets[key].cardinality() != num_samples:
                raise Exception('Cardinality {} of dataset for key {} does not match number of samples {}'.format(self.datasets[key].cardinality(), key, num_samples))
            self.datasets[key] = self.datasets[key].batch(self.gen_batch_size)
            self.datasets[key] = self.datasets[key].shuffle(self.datasets[key].cardinality(), seed=utils.get_random_seed(), reshuffle_each_iteration=True)
            if self.datasets[key].cardinality() != num_batches:
                raise Exception('Number of batches {} of dataset for key {} does not match expected number of batches {}'.format(datasets[key].cardinality(), key, num_batches))
        printl()
    def get_valid_keys(self):
        return self.valid_keys
    def get_batch_iterator(self, key: int):
        if not(type(key) == int):
            raise TypeError('key must be of type {} but has type {}'.format(int, type(key)))
        
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

def get_google_scaaml(attack_point: str,
                      num_keys: int,
                      trace_length: int,
                      byte: int,
                      plaintext_encoding: str) -> dict:
    printl('Loading google_scaaml dataset:')
    if not(type(attack_point) == str):
        raise TypeError('attack_point must be of type {} but is of type {}'.format(str, type(attack_point)))
    if not(type(num_keys) == int):
        raise TypeError('num_keys must be of type {} but is of type {}'.format(int, type(num_keys)))
    if not(type(trace_length) == int):
        raise TypeError('trace_length must be of type {} but is of type {}'.format(int, type(trace_length)))
    if not(type(byte) == int):
        raise TypeError('byte must be of type {} but is of type {}'.format(int, type(byte)))
    if not(type(plaintext_encoding) == str):
        raise TypeError('plaintext_encoding must be of type {} but is of type {}'.format(str, type(plaintext_encoding)))
    
    base_path = os.path.join(os.getcwd(), 'datasets', 'google_scaaml', 'train')
    if not(os.path.isdir(base_path)):
        raise Exception('google_scaaml dataset not found at path \'{}\''.format(path))
    printl('\tPath: \'{}\''.format(base_path))
    files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]
    printl('\tFiles found at path: {}'.format(len(files)))
    if not(0 < num_keys <= len(files)):
        raise ValueError('num_keys must be a positive integer less than the number of available keys ({}), but has value {}'.format(len(files), num_keys))
    
    keys = random.choices(files, k=num_keys)
    printl('\tKeys to be used:')
    
    data = {}
    for key in keys:
        key_int = int(key[:-4], base=16)
        path = os.path.join(base_path, key)
        printl('\t\tValue: {}'.format(hex(key_int)))
        printl('\t\t\tPath: \'{}\''.format(path))
        data.update({key_int: {}})
        shard = np.load(path)
        printl('\t\t\tEntries in shard: {}'.format(', '.join([entry for entry in shard])))
        if not('traces' in shard):
            raise Exception('traces entry missing from archive')
        traces = shard['traces']
        printl('\t\t\tTraces shape: {}'.format(traces.shape))
        if not(0 < trace_length <= traces.shape[1]):
            raise ValueError('trace_length must be a positive integer less than the available trace length ({}), but has value {}'.format(traces.shape[1], trace_length))
        traces = traces[:, :trace_length, :]
        printl('\t\t\tExtracted traces shape: {}'.format(traces.shape))
        if not('pts' in shard):
            raise Exception('pts entry missing from archive')
        available_bytes = len(shard['pts'])
        printl('\t\t\tNumber of bytes in shard: {}'.format(available_bytes))
        if not(0 <= byte < available_bytes):
            raise ValueError('byte must be a nonnegative integer less than available bytes ({}), but has value {}'.format(available_bytes, byte))
        printl('\t\t\tPlaintexts shape: {}'.format(shard['pts'].shape))
        plaintexts = shard['pts'][byte]
        printl('\t\t\tExtracted plaintexts shape: {}'.format(plaintexts.shape))
        if not(all([0 <= pt < 2**8 for pt in plaintexts])):
            raise Exception('pts entry values must be between {} and {}, but contains {}'.format(hex(0), hex(0xFF), ', '.join([hex(pt) for pt in plaintexts if not(0 <= pt < 2**8)])))
        if plaintext_encoding == 'binary':
            printl('\t\t\tEncoding plaintexts as binary vector')
            plaintexts = binary(plaintexts, 8)
        elif plaintext_encoding == 'onehot':
            printl('\t\t\tEncoding plaintexts as one hot vector')
            plaintexts = np.array(tf.one_hot(plaintexts, 256))
        elif plaintext_encoding == 'scalar':
            printl('\t\t\tLeaving plaintexts as scalars')
        else:
            raise ValueError('Invalid plaintext encoding: {}'.format(plaintext_encoding))
        printl('\t\t\tEncoded plaintexts shape: {}'.format(plaintexts.shape))
        if not(attack_point in shard):
            raise ValueError('attack point \'{}\' is not an entry in shard'.format(attack_point))
        attack_points = shard[attack_point]
        printl('\t\t\tAttack points shape: {}'.format(attack_points.shape))
        if not(len(attack_points) == available_bytes):
            raise Exception('Inconsistent amount of bytes found in shard: {} in traces entry, but {} in {} entry'.format(available_bytes, len(attack_points), attack_point))
        attack_points = attack_points[byte]
        printl('\t\t\tExtracted attack points shape: {}'.format(attack_points.shape))
        if not(all([0 <= ap < 2**8 for ap in attack_points])):
            raise Exception('{} entry values must be between {} and {}, but contains {}'.format(attack_point, hex(0), hex(2**8), ', '.join([hex(ap) for ap in attack_points if not(0 <= ap < 2**8)])))
        printl('\t\t\tEncoding attack points as one-hot vector')
        attack_points = np.array(tf.one_hot(attack_points, 256))
        printl('\t\t\tEncoded attack points shape: {}'.format(attack_points.shape))
        data[key_int].update({'traces': traces,
                              'plaintexts': plaintexts,
                              'attack_points': attack_points})
    printl()
    
    return data
    
def get_dataset(name: str,
                plaintext_encoding: str,
                num_keys: int,
                trace_length: int,
                attack_point: int,
                byte: int,
                generator_batch_size: int) -> type(Dataset):
    if not(type(name) == str):
        raise TypeError('name must be of type {} but has type {}'.format(str, type(name)))
    if not(type(plaintext_encoding) == str):
        raise TypeError('plaintext_encoding must be of type {} but has type {}'.format(str, type(plaintext_encoding)))
    if not(type(num_keys) == int):
        raise TypeError('num_keys must be of type {} but has type {}'.format(int, type(num_keys)))
    if not(type(trace_length) == int):
        raise TypeError('trace_length must be of type {} but has type {}'.format(int, type(trace_length)))
    if not(type(attack_point) == str):
        raise TypeError('attack_point must be of type {} but has type {}'.format(str, type(attack_point)))
    if not(type(byte) == int):
        raise TypeError('byte must be of type {} but has type {}'.format(int, type(byte)))
    if not(type(generator_batch_size) == int):
        raise TypeError('generator_batch_size must be of type {} but has type {}'.format(int, type(generator_batch_size)))
    
    printl('Loading data.')
    if name == 'google_scaaml':
        data = get_google_scaaml(attack_point, num_keys, trace_length, byte, plaintext_encoding)
    else:
        raise ValueError('Dataset name \'{}\' is undefined.'.format(name))
        
    printl('Constructing dataset.')
    dataset = Dataset(data, generator_batch_size)
    
    printl('Determining and verifying dimensions of dataset.')
    batch = next(dataset.get_batch_iterator(dataset.get_valid_keys()[0]))
    batch_length = len(batch)
    printl('\tBatch length: {}'.format(batch_length))
    batch_item_shapes = []
    for (idx, item) in enumerate(batch):
        batch_item_shapes.append(item.shape)
        printl('\tItem {} shape: {}'.format(idx, batch_item_shapes[-1]))
    num_batches = 0
    for key in dataset.get_valid_keys():
        for batch in dataset.get_batch_iterator(key):
            num_batches += 1
            if len(batch) != batch_length:
                raise Exception('Dataset contains batches with different lengths: {} and {}'.format(batch_length, len(batch)))
            for (idx, (item_shape, item)) in enumerate(zip(batch_item_shapes, batch)):
                if item_shape != item.shape:
                    raise Exception('Dataset contains batches with different item {} shape: {} and {}'.format(idx, item_shape, item.shape))
    printl('\tTotal number of batches: {} ({} per key)'.format(num_batches, num_batches//len(dataset.get_valid_keys())))
    printl()
    
    return dataset