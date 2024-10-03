import numpy as np
import pandas as pd
import tensorflow as tf
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
    ('--data_path', p.STR)
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


def load_pre_processed_data(nfiles, batchsize, bits):
    files = os.listdir(os.path.join(args.data_path, f'data_{bits}_eLinks'))

    train_files = [f for f in files if 'train' in f][0:nfiles]
    test_files = [f for f in files if 'test' in f][0:nfiles]

    # Load and combine all training files
    train_datasets = []
    for file in train_files:
        train_datasets.append(tf.data.experimental.load(os.path.join(args.data_path, f'data_{bits}_eLinks', file)))

    # Combine all loaded training datasets
    train_dataset = train_datasets[0]
    for ds in train_datasets[1:]:
        train_dataset = train_dataset.concatenate(ds)

    # Load and combine all test files
    test_datasets = []
    for file in test_files:
        test_datasets.append(tf.data.experimental.load(os.path.join(args.data_path, f'data_{bits}_eLinks', file)))

    # Combine all loaded test datasets
    test_dataset = test_datasets[0]
    for ds in test_datasets[1:]:
        test_dataset = test_dataset.concatenate(ds)

    print("Training dataset size:", train_dataset.cardinality().numpy())
    print("Test dataset size:", test_dataset.cardinality().numpy())

    # Prepare the data loaders
    train_loader = train_dataset.batch(batchsize)
    test_loader = test_dataset.batch(batchsize)

    # Add this mapping to format the data correctly
    train_loader = train_loader.map(lambda wafers, cond: ((wafers, cond), wafers))
    test_loader = test_loader.map(lambda wafers, cond: ((wafers, cond), wafers))

    return train_loader, test_loader



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
        all_models = [2, 3, 4, 5]
    elif args.alloc_geom == 'new':
        all_models = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
elif args.model_per_bit_config:
    if args.alloc_geom == 'old':
        all_models = [3, 5, 7, 9]
    elif args.alloc_geom == 'new':
        all_models = [1, 3, 5, 7, 9]

bitsPerOutputLink = [0, 1, 3, 5, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]

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
    def build_model(hp):
        # Hyperparameters
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='log')
        num_decoder_layers = hp.Int('num_decoder_layers', min_value=1, max_value=5)
        units_in_decoder = hp.Int('units_in_decoder', min_value=32, max_value=256, step=32)
        batch_size = hp.Int('batch_size', min_value=256, max_value=4096, step=256)
        num_epochs = hp.Int('num_epochs', min_value=20, max_value=500, step=50)
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

        for i in range(num_decoder_layers):
            units = units_in_decoder
            y = Dense(units)(y)
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

        return cae

    
    # Set up the tuner using Bayesian Optimization
    tuner = kt.BayesianOptimization(
        hypermodel=build_model,
        objective='val_loss',
        max_trials=50,  # Increase trials for better optimization
        directory='scan_output',
        project_name=model_name
    )

    # Function to get data loaders based on hyperparameters
    def get_data_loaders(hp):
        batch_size = hp.get('batch_size')
        train_loader, test_loader = load_pre_processed_data(args.num_files, batch_size, bitsPerOutput)
        return train_loader, test_loader
    fit_kwargs = {}

    # Custom callback to pass data loaders to the tuner
    class MyTuner(kt.BayesianOptimization):
        def run_trial(self, trial, *args, **kwargs):
            hp = trial.hyperparameters
            train_loader, test_loader = get_data_loaders(hp)
            # Get the number of epochs from hyperparameters
            epochs = hp.get('num_epochs')
            # Pass epochs to fit arguments
            fit_kwargs.update({'epochs': epochs})
            # Update fit arguments with data loaders
            fit_kwargs.update({'x': train_loader, 'validation_data': test_loader})
            # Call the parent run_trial and return the results
            return super(MyTuner, self).run_trial(trial, **fit_kwargs)

    tuner = MyTuner(
        hypermodel=build_model,
        objective='val_loss',
        max_trials=2,
        directory=args.opath,
        project_name=model_name
    )
    
    print('Starting search')
    # Start the hyperparameter search
    tuner.search()

    # Retrieve the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete.
    Optimal learning rate: {best_hps.get('learning_rate')}
    Optimal number of decoder layers: {best_hps.get('num_decoder_layers')}
    Optimal units in decoder layers: {best_hps.get('units_in_decoder')}
    Optimal batch size: {best_hps.get('batch_size')}
    Optimal number of epochs: {best_hps.get('num_epochs')}
    """)

    # Build the model with the optimal hyperparameters
    model = tuner.hypermodel.build(best_hps)

    # Get data loaders with optimal batch size
    train_loader, test_loader = load_pre_processed_data(args.num_files, best_hps.get('batch_size'), bitsPerOutput)

    # Define a learning rate scheduler
    def cosine_annealing(epoch, total_epochs, initial_lr):
        """Cosine annealing scheduler."""
        cos_inner = np.pi * (epoch % (total_epochs // 10))
        cos_inner /= total_epochs // 10
        cos_out = np.cos(cos_inner) + 1
        return float(initial_lr / 2 * cos_out)

    initial_lr = best_hps.get('learning_rate')
    total_epochs = best_hps.get('num_epochs')

    # Create a learning rate scheduler callback
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
