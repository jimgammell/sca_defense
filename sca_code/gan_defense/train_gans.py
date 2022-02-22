import time
from tqdm import tqdm
from models import Generator, Discriminator, System

def time_function(f):
    printl('Executing function \'%s\'...'%(f.__name__))
    printl('\tArgs: {}.'.format(args))
    printl('\tKWArgs: {}.'.format(kwargs))
    t0 = time.time()
    f(*args, **kwargs)
    printl('\tDone.')
    printl('\tTime taken: %f seconds.'%(time.time()-t0))

@time_function
def load_dataset(keys, train=True):
    pass
    
@time_function
def train_generators_step(batch, system, loss, optimizer):
    (traces, plaintexts, labels) = batch
    for generator in system.generators:
        generator.trainable = False
    system.discriminator.trainable = True
    with tf.GradientTape() as tape:
        predictions = system(traces, training=True)
        loss_value = loss(labels, predictions)
    grads = tape.gradient(loss_value, discriminator.trainable_weights)
    optimizer.apply_gradients(zip(grads, system.trainable_weights))

@time_function
def train_discriminator_step(batch, generators, discriminator, loss, optimizer):
    (traces, plaintexts, labels) = batch
    protected_traces = np.zeros_like(traces)
    for generator in generators:
        generator.trainable = False
        for (idx, (trace, label)) in enumerate(zip(traces, labels)):
            if label == generator.key:
                protected_trace[idx] = trace + generator(plaintext)
    discriminator.trainable = True
    with tf.GradientTape() as tape:
        predictions = discriminator(protected_trace, training=True)
        loss_value = loss(labels, predictions)
    grads = tape.gradient(loss_value, discriminator.trainable_weights)
    optimizer.apply_gradients(zip(grads, discriminator.trainable_weights))

@time_function
def train(dataset, generators, discriminator,
          gens_loss, disc_loss,
          gens_optimizer, disc_optimizer,
          gens_trainable=True, disc_trainable=True, num_epochs=1):
    system = System(generators, discriminator)
    for epoch in range(1, num_epochs+1):
        printl('Beginning epoch %d...'%(epoch))
        for batch in tqdm(dataset.as_numpy_iterator()):
            if disc_trainable:
                train_discriminator_step(batch, generators, discriminator, disc_loss, disc_optimizer)
            if gens_trainable:
                train_generators_step(batch, system, gens_loss, gens_optimizer)

@time_function
def evaluate(generators, discriminator):
    pass

def main():
    training_dataset = load_dataset(keys, train=True)
    test_dataset = load_dataset(keys, train=False)
    
    generators = [Generator(key) for key in keys]
    
    train_discriminator = Discriminator()
    train(generators, train_discriminator, gens_trainable=True, disc_trainable=True)
    evaluate(generator, train_discriminator)
    
    del train_discriminator
    test_discriminator = Discriminator()
    train(generators, test_discriminator, gens_trainable=False, disc_trainable=True)
    evaluate(generators, test_discriminator)

if __name__ == '__main__':
    main()