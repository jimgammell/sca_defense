from torch.utils.data import random_split, DataLoader

from dataset import AesSingleKeyDataset, AesKeyGroupDataset
import generator_models
import discriminator_models

def multigen_experiment(byte,
                        keys,
                        key_dataset_kwargs,
                        dataset_prop_for_validation,
                        dataloader_kwargs,
                        trace_map_architecture,
                        trace_map_kwargs,
                        plaintext_map_architecture,
                        plaintext_map_kwargs,
                        key_map_architecture,
                        key_map_kwargs,
                        cumulative_map_architecture,
                        cumulative_map_kwargs,
                        discriminator_architecture,
                        discriminator_kwargs,
                        discriminator_loss,
                        discriminator_loss_kwargs,
                        discriminator_optimizer,
                        discriminator_optimizer_kwargs,
                        generator_loss,
                        generator_loss_kwargs,
                        generator_optimizer,
                        generator_optimizer_kwargs):
    # Create dataset and partition into training and validation sets
    key_datasets = [AesSingleKeyDataset(byte,
                                        key,
                                        trace_transform=None,
                                        plaintext_transform=None,
                                        **key_dataset_kwargs) for key in keys]
    dataset = AesKeyGroupDataset(key_datasets, byte)
    training_dataset_length = int((1-dataset_prop_for_validation)*len(dataset))
    validation_dataset_length = int(dataset_prop_for_validation*len(dataset))
    training_dataset, validation_dataset = random_split(dataset, [training_dataset_length, validation_dataset_length])
    training_dataloader = DataLoader(training_dataset, **dataloader_kwargs)
    validation_dataloader = DataLoader(validation_dataset, **dataloader_kwargs)
    
    # Create generator and related objects
    key_to_map = {'zero': generator_models.get_zero_map,
                  'identity': generator_models.get_identity_map,
                  'mlp': generator_models.get_mlp_map,
                  'rnn': generator_models.get_rnn_map}
    trace_map_constructor = key_to_map[trace_map_architecture]
    plaintext_map_constructor = key_to_map[plaintext_map_architecture]
    key_map_constructor = key_to_map[key_map_architecture]
    cumulative_map_constructor = key_to_map[cumulative_map_architecture]
    generators = {}
    generator_optimizers = {}
    for key in keys:
        trace_map = trace_map_constructor(dataset.trace_size, dataset.trace_size, **trace_map_kwargs)
        plaintext_map = plaintext_map_constructor(dataset.plaintext_size, dataset.plaintext_size, **plaintext_map_kwargs)
        key_map = key_map_constructor(dataset.key_size, dataset.key_size, **key_map_kwargs)
        cumulative_map = cumulative_map_constructor(dataset.trace_size + dataset.plaintext_size + dataset.key_size, dataset.trace_size, **cumulative_map_kwargs)
        generator = Generator(trace_map, plaintext_map, key_map, cumulative_map)
        generators.update({key: generator})
    generator = CompositeGenerator(generators)
    generator_optimizer = generator_optimizer(**generator_optimizer_kwargs)
    generator_loss = generator_loss(**generator_loss_kwargs)
    generator_loss.reduction = 'none'
    
    # Create discriminator and related objects
    key_to_disc = {'google_style_resnet': discriminator_models.get_google_style_resnet_discriminator}
    discriminator_constructor = key_to_disc[discriminator_architecture]
    discriminator = discriminator_constructor(dataset.trace_size, **discriminator_kwargs)
    discriminator_optimizer = discriminator_optimizer(**discriminator_optimizer_kwargs)
    discriminator_loss = discriminator_loss(**discriminator_loss_kwargs)
    discriminator_loss.reduction = 'none'
    
    # Initial results
    results = Results()
    tl, ta = eval_discriminator_epoch(training_dataloader, generator, discriminator, discriminator_loss, device)
    vl, va = eval_discriminator_epoch(validation_dataloader, generator, discriminator, discriminator_loss, device)
    results.update({'discriminator': {'train_loss': tl, 'train_acc': ta, 'val_loss': vl, 'val_acc': va}})
    tl, ta = eval_generator_epoch(training_dataloader, generator, discriminator, generator_loss, device)
    vl, va = eval_generator_epoch(validation_dataloader, generator, discriminator, generator_loss, device)
    results.update({'generator': {'train_loss': tl, 'train_acc': ta, 'val_loss': vl, 'val_acc': va}})
    
    # Discriminator pretraining
    for epoch in range(discriminator_pretraining_epochs):
        tl, ta = train_discriminator_epoch(training_dataloader, generator, discriminator, discriminator_loss, discriminator_optimizer, device)
        vl, va = eval_discriminator_epoch(validation_dataloader, generator, discriminator, discriminator_loss, device)
        results.update({'discriminator': {'train_loss': tl, 'train_acc': ta, 'val_loss': vl, 'val_acc': va}})
    
    # Generator pretraining
    for epoch in range(generator_pretraining_epochs):
        tl, ta = train_generator_epoch(training_dataloader, generator, discriminator, generator_loss, generator_optimizer, device)
        vl, va = eval_generator_epoch(validation_dataloader, generator, discriminator, generator_loss, device)
        results.update({'generator': {'train_loss': tl, 'train_acc': ta, 'val_loss': vl, 'val_acc': va}})
    
    # GAN training
    for epoch in range(gan_training_epochs):
        tgl, tga, tdl, tda = train_gan_epoch(training_dataloader, generator, discriminator, generator_loss, discriminator_loss, generator_optimizer, discriminator_optimizer, device)
        vgl, vga = eval_generator_epoch(validation_dataloader, generator, discriminator, generator_loss, device)
        vdl, vda = eval_discriminator_epoch(validation_dataloader, generator, discriminator, discriminator_loss, device)
        results.update({'generator': {'train_loss': tgl, 'train_acc': tga, 'val_loss': vgl, 'val_acc': vga},
                        'discriminator': {'train_loss': tdl, 'train_acc': tda, 'val_loss': vdl, 'val_acc': vda}}
    
    return results
    