import numpy as np
import pandas as pd
import os
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

from tensorflow import keras
from keras import layers
from keras.layers import Layer
from qkeras import QActivation, QConv2D, QDense, quantized_bits
import qkeras
from qkeras.utils import model_save_quantized_weights
from keras.models import Model
from keras.layers import *
from telescope import *
from utils import *
import inspect
import json


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])  # Set limit to 2048 MB (2GB)
    except RuntimeError as e:
        print(e)

import os
import sys
import graph

import pickle
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import matplotlib.pyplot as plt
import mplhep as hep

import keras_tuner as kt  # Import Keras Tuner

p = ArgumentParser()
p.add_args(
    ('--mname', p.STR),
    ('--opath', p.STR),
    ('--mpath', p.STR), 
    ('--num_files', p.INT), ('--model_per_eLink',  p.STORE_TRUE), ('--model_per_bit_config',  p.STORE_TRUE),
    ('--alloc_geom', p.STR),
    ('--data_path', p.STR),
    ('--ft_search',  p.STORE_TRUE)
)

remap_8x8 = [4, 12, 20, 28, 5, 13, 21, 29, 6, 14, 22, 30, 7, 15, 23, 31,
             24, 25, 26, 27, 16, 17, 18, 19, 8, 9, 10, 11, 0, 1, 2, 3,
             59, 51, 43, 35, 58, 50, 42, 34, 57, 49, 41, 33, 56, 48, 40, 32]


def get_pams():
    jsonpams = {}
    opt_classes = tuple(opt[1] for opt in inspect.getmembers(tf.keras.optimizers, inspect.isclass))
    for k, v in self.pams.items():
        if type(v) == type(np.array([])):
            jsonpams[k] = v.tolist()
        elif isinstance(v, opt_classes):
            config = {}
            for hp in v.get_config():
                config[hp] = str(v.get_config()[hp])
            jsonpams[k] = config
        elif type(v) == type(telescopeMSE8x8):
            jsonpams[k] = str(v)
        else:
            jsonpams[k] = v
    return jsonpams


def load_pre_processed_data(nfiles, batchsize, cur_eLinks):
    print(f'data_{cur_eLinks}_eLinks')
    files = os.listdir(os.path.join(args.data_path, f'data_{cur_eLinks}_eLinks'))

    train_files = [f for f in files if 'train' in f][0:nfiles]
    test_files = [f for f in files if 'test' in f][0:nfiles]

    # Load and combine all training files
    train_datasets = []
    for file in train_files:
        train_datasets.append(tf.data.experimental.load(os.path.join(args.data_path, f'data_{cur_eLinks}_eLinks', file)))

    # Combine all loaded training datasets
    train_dataset = train_datasets[0]
    for ds in train_datasets[1:]:
        train_dataset = train_dataset.concatenate(ds)
        
    # Limit the training dataset to 1M samples
    train_dataset = train_dataset.take(1_000_000)


    # Load and combine all test files
    test_datasets = []
    for file in test_files:
        test_datasets.append(tf.data.experimental.load(os.path.join(args.data_path, f'data_{cur_eLinks}_eLinks', file)))

    # Combine all loaded test datasets
    test_dataset = test_datasets[0]
    for ds in test_datasets[1:]:
        test_dataset = test_dataset.concatenate(ds)
        
    # Limit the test dataset to 200k samples
    test_dataset = test_dataset.take(200_000)


    print("Training dataset size:", train_dataset.cardinality().numpy())
    print("Test dataset size:", test_dataset.cardinality().numpy())

    # Prepare the data loaders
    train_loader = train_dataset.batch(batchsize)
    test_loader = test_dataset.batch(batchsize)

    # Add this mapping to format the data correctly
    train_loader = train_loader.map(lambda wafers, cond: ((wafers, cond), wafers))
    test_loader = test_loader.map(lambda wafers, cond: ((wafers, cond), wafers))

    return train_loader, test_loader

def extract_hyperparameters(file_path):
    hyperparams = {}
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            if 'Optimal learning rate' in line:
                hyperparams['learning_rate'] = float(line.split(':')[-1].strip())
            elif 'Optimal batch size' in line:
                hyperparams['batch_size'] = int(line.split(':')[-1].strip())
            elif 'Optimal number of epochs' in line:
                hyperparams['num_epochs'] = int(line.split(':')[-1].strip())
            elif 'Optimal LR scheduler' in line:
                hyperparams['lr_sched'] = line.split(':')[-1].strip()
            elif 'Best performance' in line:
                hyperparams['best_performance'] = float(line.split(':')[-1].strip())
    print('Loaded hyperparameters')
    print(hyperparams)
    return hyperparams


class keras_pad(Layer):
    def call(self, x):
        padding = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
        return tf.pad(
            x, padding, mode='CONSTANT', constant_values=0, name=None
        )


class keras_minimum(Layer):
    def call(self, x, sat_val=1):
        return tf.minimum(x, sat_val)


class keras_floor(Layer):
    def call(self, x):
        if isinstance(x, tf.SparseTensor):
            x = tf.sparse.to_dense(x)

        return tf.math.floor(x)


args = p.parse_args()
model_dir = args.opath

if not os.path.exists(model_dir):
    os.system("mkdir -p " + model_dir)

# Loop through each number of eLinks

if args.model_per_eLink:
    if args.alloc_geom == 'old':
#         all_models = [2,3]
#         all_models = [4,5]
        all_models = [4,5]

#         all_models = [2,3]

    elif args.alloc_geom == 'new':
        all_models = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
elif args.model_per_bit_config:
    if args.alloc_geom == 'old':
        all_models = [3, 5, 7, 9]
    elif args.alloc_geom == 'new':
        all_models = [1, 3, 5, 7, 9]

bitsPerOutputLink = [0, 1, 3, 5, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]

print(all_models)
for m in all_models:
    if args.model_per_eLink:
        eLinks = m
        bitsPerOutput = bitsPerOutputLink[eLinks]
        print(f'Training Model with {eLinks} eLinks')
        model_dir = os.path.join(args.opath, f'model_{eLinks}_eLinks')
        model_name = f'model_{eLinks}_eLinks'
        
    elif args.model_per_bit_config:
        bitsPerOutput = m
        print(f'Training Model with {bitsPerOutput} output bits')
        model_dir = os.path.join(args.opath, f'model_{bitsPerOutput}_bits')
        model_name = f'model_{bitsPerOutput}_bits'
        
    print(m)
    if not os.path.exists(model_dir):
        os.system("mkdir -p " + model_dir)

    nIntegerBits = 1
    nDecimalBits = bitsPerOutput - nIntegerBits
    outputSaturationValue = (1 << nIntegerBits) - 1./(1 << nDecimalBits)
    maxBitsPerOutput = 9
    outputMaxIntSize = 1

    if bitsPerOutput > 0:
        outputMaxIntSize = 1 << nDecimalBits

    outputMaxIntSizeGlobal = 1
    if maxBitsPerOutput > 0:
        outputMaxIntSizeGlobal = 1 << (maxBitsPerOutput - nIntegerBits)

    # Set up hyperparameter tuning
    def build_model(hp, enc_dec = False):
        # Hyperparameters
        if args.ft_search:
            file_path = os.path.join(args.opath,model_name,'hyperparameter_search_results.txt')
            best_hyperparams = extract_hyperparameters(file_path)
            learning_rate = hp.Float('learning_rate', 
                                     min_value=best_hyperparams['learning_rate']/2, 
                                     max_value=best_hyperparams['learning_rate']*2, 
                                     sampling='log')
            batch_size = hp.Int('batch_size', 
                                min_value=int(best_hyperparams['batch_size']/2), 
                                max_value=best_hyperparams['batch_size']*2, 
                                step=int(best_hyperparams['batch_size']/16))
            num_epochs = hp.Int('num_epochs', 
                                min_value=int(best_hyperparams['num_epochs']*0.75), 
                                max_value=best_hyperparams['num_epochs']*2, 
                                step=50)
            lr_scheduler = hp.Choice('lr_sched', values=[best_hyperparams['lr_sched']])
        else:
            learning_rate = hp.Float('learning_rate', min_value=1e-6, max_value=1e-3, sampling='log')
            batch_size = hp.Int('batch_size', min_value=128, max_value=4096, step=256)
            num_epochs = hp.Int('num_epochs', min_value=50, max_value=250, step=50)
            lr_scheduler = hp.Choice('lr_sched', values=['cos', 'cos_warm_restarts'])

        #         num_epochs = hp.Int('num_epochs', min_value=11, max_value=13, step=1)

        # Fixed parameters
        n_kernels = 8
        n_encoded = 16
        conv_weightBits = 6
        conv_biasBits = 6
        dense_weightBits = 6
        dense_biasBits = 6
        encodedBits = 9
        CNN_kernel_size = 3

        # Encoder
        input_enc = Input(shape=(8, 8, 1), name='Wafer')
        cond = Input(shape=(8,), name='Cond')

        # Quantizing input, 8 bit quantization, 1 bit for integer
        x = QActivation(quantized_bits(bits=8, integer=1), name='input_quantization')(input_enc)
        x = keras_pad()(x)
        x = QConv2D(n_kernels,
                    CNN_kernel_size,
                    strides=2, padding='valid',
                    kernel_quantizer=quantized_bits(bits=conv_weightBits, integer=0, keep_negative=1, alpha=1),
                    bias_quantizer=quantized_bits(bits=conv_biasBits, integer=0, keep_negative=1, alpha=1),
                    name="conv2d")(x)
        x = QActivation(quantized_bits(bits=8, integer=1), name='act')(x)
        x = Flatten()(x)
        x = QDense(n_encoded,
                   kernel_quantizer=quantized_bits(bits=dense_weightBits, integer=0, keep_negative=1, alpha=1),
                   bias_quantizer=quantized_bits(bits=dense_biasBits, integer=0, keep_negative=1, alpha=1),
                   name="dense")(x)
        # Quantizing latent space, 9 bit quantization, 1 bit for integer
        x = QActivation(qkeras.quantized_bits(bits=9, integer=1), name='latent_quantization')(x)
        latent = x
        if bitsPerOutput > 0 and maxBitsPerOutput > 0:
            latent = keras_floor()(latent * outputMaxIntSize)
            latent = keras_minimum()(latent / outputMaxIntSize, sat_val=outputSaturationValue)

        latent = concatenate([latent, cond], axis=1)

        encoder = keras.Model([input_enc, cond], latent, name="encoder")

        # Decoder
        input_dec = Input(shape=(24,))
        y = input_dec
        y = Dense(24)(input_dec)
        y = ReLU()(y)
        y = Dense(64)(y)
        y = ReLU()(y)
        y = Dense(128)(y)
        y = ReLU()(y)
        y = Reshape((4, 4, 8))(y)
        y = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='valid')(y)
        y = y[:, 0:8, 0:8]
        y = ReLU()(y)
        recon = y

        decoder = keras.Model([input_dec], recon, name="decoder")

        # Combined model
        cae = Model(
            inputs=[input_enc, cond],
            outputs=decoder([encoder([input_enc, cond])]),
            name="cae"
        )

        # Compile the model
        loss = telescopeMSE8x8
        opt = tf.keras.optimizers.Lion(learning_rate=learning_rate, weight_decay=0.00025)

        cae.compile(optimizer=opt, loss=loss)
        if enc_dec:
            return cae, encoder, decoder
        else:
            return cae
    
    # Set up the tuner using Bayesian Optimization
    tuner = kt.BayesianOptimization(
        hypermodel=build_model,
        objective='val_loss',
        max_trials=20,  # Increase trials for better optimization
        directory='scan_output',
        project_name=model_name
    )

    # Function to get data loaders based on hyperparameters
    def get_data_loaders(hp):
        batch_size = hp.get('batch_size')
        print('Loading')
        print(eLinks)
        train_loader, test_loader = load_pre_processed_data(args.num_files, batch_size, eLinks)
        return train_loader, test_loader
    fit_kwargs = {}

    # Define learning rate scheduling functions
    def cos_warm_restarts(epoch, total_epochs, initial_lr):
        """Cosine annealing scheduler with warm restarts."""
        cos_inner = np.pi * (epoch % (total_epochs // 25))
        cos_inner /= total_epochs // 25
        cos_out = np.cos(cos_inner) + 1
        return float(initial_lr / 2 * cos_out)

    def cosine_annealing(epoch, total_epochs, initial_lr):
        """Cosine annealing scheduler that reduces the learning rate to 1/100 of the initial value."""
        cos_inner = np.pi * (epoch % total_epochs) / total_epochs
        cos_out = np.cos(cos_inner) + 1
        return float((initial_lr / 2) * cos_out * (1 / 100))


    class MyTuner(kt.BayesianOptimization):
        def run_trial(self, trial, **kwargs):
            hp = trial.hyperparameters
            train_loader, test_loader = get_data_loaders(hp)

            # Get the number of epochs and learning rate from hyperparameters
            epochs = hp.get('num_epochs')
            initial_lr = hp.get('learning_rate')
            lr_sched = hp.get('lr_sched')
            if lr_sched == 'cos_warm_restarts':
                lr_schedule = lambda epoch: cos_warm_restarts(epoch, total_epochs=epochs, initial_lr=initial_lr)
            elif lr_sched == 'cos':
                lr_schedule = lambda epoch: cosine_annealing(epoch, total_epochs=epochs, initial_lr=initial_lr)

            # Define the LearningRateScheduler callback with the dynamically updated lr_schedule
            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

            # Update fit arguments with data loaders and callbacks
            fit_kwargs.update({
                'x': train_loader,
                'validation_data': test_loader,
                'epochs': epochs,
                'callbacks': [lr_scheduler]  # Adding the LR scheduler here
            })

            # Call the parent run_trial and return the results
            return super(MyTuner, self).run_trial(trial, **fit_kwargs)

    
    if args.ft_search:
        trials = 10
        opath = args.opath+'_ft'
    else:
        trials = 20
        opath = args.opath
    
    tuner = MyTuner(
        hypermodel=build_model,
        objective='val_loss',
        max_trials=trials,
        directory=opath,
        project_name=model_name
    )
    
    from tensorflow.keras.callbacks import ModelCheckpoint
    checkpoint_cb = ModelCheckpoint(
    filepath='best-model.tf',
    save_best_only=True,  # Only save the best model
    monitor='val_loss',   # Monitor validation loss for the best model
    mode='min'            # Save when val_loss is minimized
    )
    print('Starting search')
    # Start the hyperparameter search
    tuner.search(callbacks=[checkpoint_cb])

    # Retrieve the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    best_performance = best_trial.score  # This retrieves the best score (e.g., validation loss)

    print(f"""
    The hyperparameter search is complete.
    Optimal learning rate: {best_hps.get('learning_rate')}
    Optimal batch size: {best_hps.get('batch_size')}
    Optimal number of epochs: {best_hps.get('num_epochs')}
    Optimal LR scheduler: {best_hps.get('lr_sched')}
    Best performance: {best_performance}
    """)
    content = f"""
    The hyperparameter search is complete.
    Optimal learning rate: {best_hps.get('learning_rate')}
    Optimal batch size: {best_hps.get('batch_size')}
    Optimal number of epochs: {best_hps.get('num_epochs')}
    Optimal LR scheduler: {best_hps.get('lr_sched')}
    Best performance: {best_performance}
    """
    file_path = os.path.join(model_dir, "hyperparameter_search_results.txt")
    with open(file_path, "w") as file:
        file.write(content)

    # Build the model with the optimal hyperparameters
    model, encoder, decoder = tuner.hypermodel.build(best_hps,enc_dec = True)

    # Get data loaders with optimal batch size
    train_loader, test_loader = load_pre_processed_data(args.num_files, best_hps.get('batch_size'), m)


    initial_lr = best_hps.get('learning_rate')
    total_epochs = best_hps.get('num_epochs')
    lr_sched = best_hps.get('lr_sched')

    
    if lr_sched == 'cos_warm_restarts':

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: cos_warm_restarts(epoch, total_epochs, initial_lr)
        )

    elif lr_sched == 'cos':
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: cosine_annealing(epoch, total_epochs, initial_lr)
        )

    # Train the model with the optimal hyperparameters
    history = model.fit(
        train_loader,
        validation_data=test_loader,
        epochs=total_epochs,
        callbacks=[lr_scheduler]
    )
    
    # Save the model
    
    model.save(os.path.join(model_dir, 'best_model.h5'))
    encoder.save_weights(os.path.join(model_dir, 'best-encoder-epoch.tf'))
    decoder.save_weights(os.path.join(model_dir, 'best-decoder-epoch.tf'))
