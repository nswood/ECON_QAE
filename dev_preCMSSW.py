import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from qkeras import QActivation,QConv2D,QDense,quantized_bits
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


p = ArgumentParser()
p.add_args(
    ('--mname', p.STR),
    ('--mpath', p.STR), 
    ('--model_per_eLink',  p.STORE_TRUE),
    ('--model_per_bit_config',  p.STORE_TRUE),
    ('--alloc_geom', p.STR),
    ('--load_from_scan', p.STORE_TRUE),
    ('--specific_m',  p.INT),
     
)

    
remap_8x8 = [ 4, 12, 20, 28,  5, 13, 21, 29,  6, 14, 22, 30,  7, 15, 23, 31, 
              24, 25, 26, 27, 16, 17, 18, 19,  8,  9, 10, 11,  0,  1,  2,  3, 
              59, 51, 43, 35, 58, 50, 42, 34, 57, 49, 41, 33, 56, 48, 40, 32]


def get_pams():
    jsonpams={}      
    opt_classes = tuple(opt[1] for opt in inspect.getmembers(tf.keras.optimizers,inspect.isclass))
    for k,v in self.pams.items():
        if type(v)==type(np.array([])):
            jsonpams[k] = v.tolist()
        elif  isinstance(v,opt_classes):
            config = {}
            for hp in v.get_config():
                config[hp] = str(v.get_config()[hp])
            jsonpams[k] = config
        elif  type(v)==type(telescopeMSE8x8):
            jsonpams[k] =str(v) 
        else:
            jsonpams[k] = v 
    return jsonpams


def save_models(encoder,decoder, name, isQK=False):

    f'./{model_dir}/{name}.json'
    if isQK:
        encoder_qWeight = model_save_quantized_weights(encoder)
        with open(f'{model_dir}/encoder_{name}.pkl','wb') as f:
            pickle.dump(encoder_qWeight,f)
    graph.write_frozen_dummy_enc(encoder,'encoder_'+name+'.pb',logdir = model_dir)
    graph.write_frozen_graph_dec(decoder,'decoder_'+name+'.pb',logdir = model_dir)

    

def load_matching_state_dict(model, state_dict_path):
    state_dict = tf.compat.v1.train.load_checkpoint(state_dict_path)
    model_variables = model.trainable_variables
    filtered_state_dict = {}
    for var in model_variables:
        var_name = var.name.split(':')[0]
        if var_name in state_dict:
            filtered_state_dict[var_name] = state_dict[var_name]
    tf.compat.v1.train.init_from_checkpoint(state_dict_path, filtered_state_dict)

args = p.parse_args()
model_dir = args.mpath + '_CMSSW'

if not os.path.exists(model_dir):
    os.system("mkdir -p "+model_dir)
    
    
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
        model_dir = os.path.join(args.mpath+'_CMSSW', f'model_{eLinks}_eLinks')
    elif args.model_per_bit_config:
        bitsPerOutput = m
        print(f'Training Model with {bitsPerOutput} output bits')
        model_dir = os.path.join(args.mpath+'_CMSSW', f'model_{bitsPerOutput}_bits')
    
    if not os.path.exists(model_dir):
        os.system("mkdir -p " + model_dir)
    nIntegerBits = 1;
    nDecimalBits = bitsPerOutput - nIntegerBits;
    outputSaturationValue = (1 << nIntegerBits) - 1./(1 << nDecimalBits);
    maxBitsPerOutput = 9
    outputMaxIntSize = 1

    if bitsPerOutput > 0:
        outputMaxIntSize = 1 << nDecimalBits

    outputMaxIntSizeGlobal = 1
    if maxBitsPerOutput > 0:
        outputMaxIntSizeGlobal = 1 << (maxBitsPerOutput - nIntegerBits)

    batch = 1

    n_kernels = 8
    n_encoded=16
    conv_weightBits  = 6 
    conv_biasBits  = 6 
    dense_weightBits  = 6 
    dense_biasBits  = 6 
    encodedBits = 9
    CNN_kernel_size = 3
    padding = tf.constant([[0,0],[0, 1], [0, 1], [0, 0]])


    input_enc = Input(batch_shape=(batch,8,8,1), name = 'Wafer')

    # Quantizing input, 8 bit quantization, 1 bit for integer
    x = QActivation(quantized_bits(bits = 8, integer = 1),name = 'input_quantization')(input_enc)
    x = tf.pad(
        x, padding, mode='CONSTANT', constant_values=0, name=None
    )
    x = QConv2D(n_kernels,
                CNN_kernel_size, 
                strides=2,padding = 'valid', kernel_quantizer=quantized_bits(bits=conv_weightBits,integer=0,keep_negative=1,alpha=1), bias_quantizer=quantized_bits(bits=conv_biasBits,integer=0,keep_negative=1,alpha=1),
                name="conv2d")(x)

    x = QActivation(quantized_bits(bits = 8, integer = 1),name = 'act')(x)

    # x = QActivation("quantized_relu(bits=8,integer=1)", name="act")(x)

    x = Flatten()(x)

    x = QDense(n_encoded, 
               kernel_quantizer=quantized_bits(bits=dense_weightBits,integer=0,keep_negative=1,alpha=1),
               bias_quantizer=quantized_bits(bits=dense_biasBits,integer=0,keep_negative=1,alpha=1),
               name="dense")(x)

    # Quantizing latent space, 9 bit quantization, 1 bit for integer
    x = QActivation(qkeras.quantized_bits(bits = 9, integer = 1),name = 'latent_quantization')(x)

#     x = concatenate([x,bottom_row],axis=1)
   

    latent = x
    if bitsPerOutput > 0 and maxBitsPerOutput > 0:
        latent = tf.minimum(tf.math.floor(latent *  outputMaxIntSize) /  outputMaxIntSize, outputSaturationValue)

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
    
    encoder = keras.Model([input_enc], latent, name="encoder")
    decoder = keras.Model([input_dec], recon, name="decoder")


    if args.load_from_scan:
        if args.model_per_eLink:
            model_path = os.path.join(model_dir.split('_CMSSW')[0],f'model_{m}_eLinks', 'best_model.h5')
        elif args.model_per_bit_config:
            model_path = os.path.join(model_dir.split('_CMSSW')[0],f'model_{m}_eLinks', 'best_model.h5')
        encoder.load_weights(model_path, by_name=True, skip_mismatch=True)
        decoder.load_weights(model_path, by_name=True, skip_mismatch=True)

    else:

        if args.model_per_eLink:
            encoder_path = os.path.join(args.mpath,f'model_{m}_eLinks','best-encoder-epoch.tf')
            decoder_path = os.path.join(args.mpath,f'model_{m}_eLinks','best-decoder-epoch.tf')

        elif args.model_per_bit_config:
            encoder_path = os.path.join(args.mpath,f'model_{m}_bits','best-encoder-epoch.tf')
            decoder_path = os.path.join(args.mpath,f'model_{m}_bits','best-decoder-epoch.tf')


        encoder.load_weights(encoder_path)

        decoder.load_weights(decoder_path)
    
    loss = telescopeMSE8x8
    
    opt = tf.keras.optimizers.Adam(learning_rate = 0.1,weight_decay = 0.000025)

    print('loaded model')
    save_models(encoder,decoder,args.mname,isQK = True)

