import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Layer
from qkeras import QActivation,QConv2D,QDense,quantized_bits
import qkeras
from qkeras.utils import model_save_quantized_weights
from keras.models import Model
from keras.layers import *
from telescope import *
from utils import *
import inspect
import json

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

from files import get_rootfiles
from coffea.nanoevents import NanoEventsFactory
import awkward as ak
import numpy as np

import os
import sys
import graph

import pickle
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import matplotlib.pyplot as plt
import mplhep as hep

p = ArgumentParser()
p.add_args(
 
    # Paths
    ('--opath', p.STR),
    ('--mpath', p.STR),

    # Model parameters
    ('--mname', p.STR),
    ('--model_per_eLink',  p.STORE_TRUE,),
    ('--model_per_bit_config',  p.STORE_TRUE),
    ('--alloc_geom', p.STR),
    ('--specific_m',  p.INT),
    
    # Training parameters
    ('--continue_training', p.STORE_TRUE), 
    ('--loss', p.STR),
    ('--lr', {'type': float}),
    ('--nepochs', p.INT),
    ('--batchsize', p.INT),
    ('--optim', p.STR, 'lion'),
    ('--lr_scheduler', p.STR, 'cos'),

    # Dataset parameters
    ('--data_path', p.STR),
    ('--num_files', p.INT),
    ('--biased', {'type': float}), 
    ('--train_dataset_size',  p.INT, 500000), 
    ('--val_dataset_size',  p.INT, 100000), 
    ('--test_dataset_size',  p.INT, 100000),

)

remap_8x8 = [ 4, 12, 20, 28,  5, 13, 21, 29,  6, 14, 22, 30,  7, 15, 23, 31, 
              24, 25, 26, 27, 16, 17, 18, 19,  8,  9, 10, 11,  0,  1,  2,  3, 
              59, 51, 43, 35, 58, 50, 42, 34, 57, 49, 41, 33, 56, 48, 40, 32]


with open('eLink_filts.pkl', 'rb') as f:
    key_df = pickle.load(f)

    

class keras_pad(Layer):
    def call(self, x):
        padding = tf.constant([[0,0],[0, 1], [0, 1], [0, 0]])
        return tf.pad(
        x, padding, mode='CONSTANT', constant_values=0, name=None
    )
    
class keras_minimum(Layer):
    def call(self, x, sat_val = 1):
        return tf.minimum(x,sat_val)
    
class keras_floor(Layer):
    def call(self, x):
        if isinstance(x, tf.SparseTensor):
            x = tf.sparse.to_dense(x)
            
        return tf.math.floor(x)
          
args = p.parse_args()
model_dir = args.opath

if not os.path.exists(model_dir):
    os.system("mkdir -p "+model_dir)


# Loop through each number of eLinks
if args.specific_m is not None:
    all_models = [args.specific_m]
elif args.model_per_eLink:
    if args.alloc_geom == 'old':
        all_models = [2,3,4,5]
    elif args.alloc_geom =='new':
        all_models = [1,2,3,4,5,6,7,8,9,10,11]
elif args.model_per_bit_config:
    if args.alloc_geom == 'old':
        all_models = [3,5,7,9]
    elif args.alloc_geom =='new':
        all_models = [1,3,5,7,9]

bitsPerOutputLink = [0, 1, 3, 5, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]    

for m in all_models:
    if args.model_per_eLink:
        eLinks = m
        bitsPerOutput = bitsPerOutputLink[eLinks]
        print(f'Training Model with {eLinks} eLinks')
        model_dir = os.path.join(args.opath, f'model_{eLinks}_eLinks')
    elif args.model_per_bit_config:
        bitsPerOutput = m
        print(f'Training Model with {bitsPerOutput} output bits')
        model_dir = os.path.join(args.opath, f'model_{bitsPerOutput}_bits')
    
    if not os.path.exists(model_dir):
        os.system("mkdir -p " + model_dir)
    
    # Model Parameters
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

    batch = args.batchsize
    n_kernels = 8
    n_encoded=16
    conv_weightBits  = 6 
    conv_biasBits  = 6 
    dense_weightBits  = 6 
    dense_biasBits  = 6 
    encodedBits = 9
    CNN_kernel_size = 3
    

    # Inputs
    input_enc = Input(batch_shape=(batch,8,8, 1), name = 'Wafer')    
    cond = Input(batch_shape=(batch, 8), name = 'Cond')


    # Quantizing input, 8 bit quantization, 1 bit for integer
    x = QActivation(quantized_bits(bits = 8, integer = 1),name = 'input_quantization')(input_enc)
    
    x = keras_pad()(x)
    
    x = QConv2D(n_kernels,
                CNN_kernel_size, 
                strides=2,padding = 'valid', kernel_quantizer=quantized_bits(bits=conv_weightBits,integer=0,keep_negative=1,alpha=1), bias_quantizer=quantized_bits(bits=conv_biasBits,integer=0,keep_negative=1,alpha=1),
                name="conv2d")(x)
    
    x = QActivation(quantized_bits(bits = 8, integer = 1),name = 'act')(x)

    x = Flatten()(x)

    x = QDense(n_encoded, 
               kernel_quantizer=quantized_bits(bits=dense_weightBits,integer=0,keep_negative=1,alpha=1),
               bias_quantizer=quantized_bits(bits=dense_biasBits,integer=0,keep_negative=1,alpha=1),
               name="dense")(x)

    # Quantizing latent space, 9 bit quantization, 1 bit for integer
    latent = QActivation(qkeras.quantized_bits(bits = 9, integer = 1),name = 'latent_quantization')(x)

    if bitsPerOutput > 0 and maxBitsPerOutput > 0:
        latent = keras_floor()(latent *  outputMaxIntSize)
        latent = keras_minimum()(latent/outputMaxIntSize, sat_val = outputSaturationValue)

    latent = concatenate([latent,cond],axis=1)
    
    input_dec = Input(batch_shape=(batch,24))
    y = Dense(24)(input_dec)
    y = ReLU()(y)
    y = Dense(64)(y)
    y = ReLU()(y)
    y = Dense(128)(y)
    y = ReLU()(y)
    y = Reshape((4, 4, 8))(y)
    y = Conv2DTranspose(1, (3, 3), strides=(2, 2),padding = 'valid')(y)
    y =y[:,0:8,0:8]
    y = ReLU()(y)
    recon = y

    encoder = keras.Model([input_enc,cond], latent, name="encoder")
    decoder = keras.Model([input_dec], recon, name="decoder")

    cae = Model(
        inputs=[input_enc,cond],
        outputs=decoder([encoder([input_enc,cond])]),
        name="cae"
    )

    if args.loss == 'mse':
        loss=mean_mse_loss
    elif args.loss == 'tele':
        print('Using tele')
        loss = telescopeMSE8x8
    elif args.loss == 'emd':
        loss = get_emd_loss(args.emd_pth)

    if args.optim == 'adam':
        print('Using ADAM Optimizer')
        opt = tf.keras.optimizers.Adam(learning_rate = args.lr,weight_decay = 0.000025)
    elif args.optim == 'lion':
        print('Using Lion Optimizer')
        opt = tf.keras.optimizers.Lion(learning_rate = args.lr,weight_decay = 0.00025)

    cae.compile(optimizer=opt, loss=loss)
    cae.summary()


    initial_lr = args.lr
    total_epochs = args.nepochs
    
    if args.lr_scheduler == 'cos_warm_restarts':
        lr_schedule = lambda epoch: cos_warm_restarts(epoch, total_epochs=epochs, initial_lr=initial_lr)
    elif args.lr_scheduler == 'cos':
        lr_schedule = lambda epoch: cosine_annealing(epoch, total_epochs=epochs, initial_lr=initial_lr)
    
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    print(f'Training with {args.lr_scheduler} scheduler')
    
    
    # Loading Model
    if args.continue_training:
        cae.load_weights(os.path.join(model_dir, 'best-epoch.tf'))
        start_epoch = 1
        print(f"Continuing training from best model...")


    print('Loading Data')
    train_loader, test_loader = load_pre_processed_data(args.num_files,batch,m)
    print('Data Loaded')
    
    
    best_val_loss = 1e9
    all_train_loss = []
    all_val_loss = []
    all_mae = []
    all_percent_error = []
    all_sum_error = []

    if args.continue_training:
        loss_dict = {'train_loss': pd.read_csv(os.path.join(model_dir,'df.csv'))['train_loss'].tolist(), 
        'val_loss': pd.read_csv(os.path.join(model_dir,'df.csv'))['val_loss'].tolist()}
        start_epoch = 1
    else:
        start_epoch = 1
        loss_dict = {'train_loss': [], 'val_loss': [], 'mae':[], 'percent_error':[],'sum_error':[]}


    for epoch in range(start_epoch, args.nepochs):
        
        new_lr = cosine_annealing(epoch, total_epochs, initial_lr)
        tf.keras.backend.set_value(opt.learning_rate, new_lr)

        total_loss_train = 0
        for wafers, cond in train_loader:

            loss = cae.train_on_batch([wafers,cond], wafers)
            total_loss_train = total_loss_train + loss


        total_loss_val = 0 
        for wafers, cond in test_loader:

            loss = cae.test_on_batch([wafers, cond], wafers)
            total_loss_val = total_loss_val+loss




        total_loss_train = total_loss_train /(len(train_loader))
        total_loss_val = total_loss_val /(len(test_loader))
        print('Epoch {:03d}, Loss: {:.8f}, ValLoss: {:.8f}'.format(
                epoch, total_loss_train,  total_loss_val))

        loss_dict['train_loss'].append(total_loss_train)
        loss_dict['val_loss'].append(total_loss_val)

        df = pd.DataFrame.from_dict(loss_dict)

        plt.figure(figsize=(10, 6))
        plt.plot(df['train_loss'], label='Training Loss')
        plt.plot(df['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)

        # Saving the plot in the same directory as the loss CSV
        plot_path = f"{model_dir}/training_loss_plot.png"
        plt.savefig(plot_path)
        plt.close()

        df.to_csv(f"{model_dir}/df.csv", index=False)

        if total_loss_val < best_val_loss:
            print('New Best Model')
            best_val_loss = total_loss_val
            cae.save_weights(os.path.join(model_dir, 'best-epoch.tf'.format(epoch)))
            encoder.save_weights(os.path.join(model_dir, 'best-encoder-epoch.tf'.format(epoch)))
            decoder.save_weights(os.path.join(model_dir, 'best-decoder-epoch.tf'.format(epoch)))
    
    # Testing model on test dataset

    # Loading best-epoch.tf

    save_models(cae,args.mname,isQK = True)
    
    

import subprocess

if args.model_per_eLink:
    args = [
        'python', 'dev_preCMSSW.py',
        '--mname', 'vanilla_AE',
        '--mpath', args.opath,
        '--model_per_eLink',
        '--alloc_geom', args.alloc_geom, 
        '--specific_m', args.specific_m
        ]
elif args.model_per_bit_config:
    args = [
        'python', 'dev_preCMSSW.py',
        '--mname', 'vanilla_AE',
        '--mpath', args.opath,
        '--model_per_bit_config',
        '--alloc_geom', args.alloc_geom,
        '--specific_m', args.specific_m
        ]

# Run the other script with the arguments
subprocess.run(args)
