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
    ('--loss', p.STR), ('--nepochs', p.INT),
    ('--opath', p.STR),
    ('--mpath', p.STR),('--prepath', p.STR),('--continue_training', p.STORE_TRUE), ('--batchsize', p.INT),
    ('--lr', {'type': float}),
    ('--num_files', p.INT),('--pretrain_model', p.STORE_TRUE),('--optim', p.STR)
    
    
    
)

# remap_9x9 = [
#     4, 13, 22, 31, 5, 14, 23, 32, 6, 15, 24, 33, 7, 16, 25, 34,
#     27, 28, 29, 30, 18, 19, 20, 21, 9, 10, 11, 12, 0, 1, 2, 3,
#     66, 57, 48, 40, 65, 56, 50, 49, 64, 60, 59, 58, 70, 69, 68, 67
# ]

# remap_9x9_matrix = np.zeros(48*81,dtype=np.float32).reshape((81,48))

# for i in range(48): 
#     remap_9x9_matrix[remap_9x9[i],i] = 1

args = p.parse_args()
model_dir = args.opath
if not os.path.exists(model_dir):
    os.system("mkdir -p "+model_dir)

batch = 1

n_kernels = 8
n_encoded=16
conv_weightBits  = 6 
conv_biasBits  = 6 
dense_weightBits  = 6 
dense_biasBits  = 6 
encodedBits = 9
CNN_kernel_size = 3

input_enc = Input(batch_shape=(batch,9,8,1))
input_8_8 = input_enc[:,0:8]
input_cond = tf.expand_dims(tf.squeeze(input_enc[:, 8]),axis = 0)
# input_cond = Reshape((1, 8))(input_cond)# sum_input quantization is done in the dataloading step for simplicity
# cond = Input(batch_shape=(batch,2),)
# eta = Input(batch_shape =(batch,1))

# Quantizing input, 8 bit quantization, 1 bit for integer
x = QActivation(quantized_bits(bits = 8, integer = 1),name = 'input_quantization')(input_8_8)


# Check padding
x = QConv2D(n_kernels,
            CNN_kernel_size, 
            strides=2,padding = 'same', kernel_quantizer=quantized_bits(bits=conv_weightBits,integer=0, keep_negative=1,alpha=1), bias_quantizer=quantized_bits(bits=conv_biasBits,integer=0,keep_negative=1,alpha=1),
            name="conv2d")(x)
print(x.shape)
x = QActivation(quantized_bits(bits = 8, integer = 1),name = 'act')(x)

# x = QActivation("quantized_relu(bits=8,integer=1)", name="act")(x)

x = Flatten()(x)

x = QDense(n_encoded, 
           kernel_quantizer=quantized_bits(bits=dense_weightBits,integer=0,keep_negative=1,alpha=1),
           bias_quantizer=quantized_bits(bits=dense_biasBits,integer=0,keep_negative=1,alpha=1),
           name="dense")(x)

# Quantizing latent space, 9 bit quantization, 1 bit for integer
x = QActivation(qkeras.quantized_bits(bits = 9, integer = 1),name = 'latent_quantization')(x)
x = concatenate([x,input_cond],axis=1)

latent = x

input_dec = Input(batch_shape=(batch,24))
y = Dense(24)(input_dec)
y = ReLU()(y)
y = Dense(64)(y)
y = ReLU()(y)
y = Dense(128)(y)
y = ReLU()(y)
y = Reshape((4, 4, 8))(y)
y = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same')(y)
y = ReLU()(y)
recon = y

encoder = keras.Model([input_enc], latent, name="encoder")
decoder = keras.Model([input_dec], recon, name="decoder")

cae = Model(
    inputs=[input_enc],
    outputs=decoder([encoder([input_enc])]),
    name="cae"
)

if args.loss == 'mse':
    loss=mean_mse_loss
elif args.loss == 'tele':
    loss = telescopeMSE9x9
print(args.optim)
if args.optim == 'adam':
    print('Using ADAM Optimizer')
    opt = tf.keras.optimizers.Adam(learning_rate = args.lr,weight_decay = 0.000025)
elif args.optim == 'lion':
    print('Using Lion Optimizer')
    opt = tf.keras.optimizers.Lion(learning_rate = args.lr,weight_decay = 0.00025)
    
cae.compile(optimizer=opt, loss=loss)
cae.summary()

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
        elif  type(v)==type(telescopeMSE9x9):
            jsonpams[k] =str(v) 
        else:
            jsonpams[k] = v 
    return jsonpams
def save_models(autoencoder, name, isQK=False):
    
    #fix all this saving shit
    

    json_string = autoencoder.to_json()
    encoder = autoencoder.get_layer("encoder")
    decoder = autoencoder.get_layer("decoder")
    f'./{model_dir}/{name}.json'
    with open(f'./{model_dir}/{name}.json','w') as f:        f.write(autoencoder.to_json())
    with open(f'./{model_dir}/encoder_{name}.json','w') as f:            f.write(encoder.to_json())
    with open(f'./{model_dir}/decoder_{name}.json','w') as f:            f.write(decoder.to_json())

    autoencoder.save_weights(f'./{model_dir}/{name}.hdf5')
    encoder.save_weights(f'./{model_dir}/encoder_{name}.hdf5')
    decoder.save_weights(f'./{model_dir}/decoder_{name}.hdf5')
    if isQK:
        encoder_qWeight = model_save_quantized_weights(encoder)
        with open(f'{model_dir}/encoder_{name}.pkl','wb') as f:
            pickle.dump(encoder_qWeight,f)
        encoder = graph.set_quantized_weights(encoder,f'{model_dir}/encoder_'+name+'.pkl')
    graph.write_frozen_dummy_enc(encoder,'encoder_'+name+'.pb',logdir = model_dir)
    graph.write_frozen_dummy_enc(encoder,'encoder_'+name+'.pb.ascii',logdir = model_dir,asText=True)
    graph.write_frozen_graph_dec(decoder,'decoder_'+name+'.pb',logdir = model_dir)
    graph.write_frozen_graph_dec(decoder,'decoder_'+name+'.pb.ascii',logdir = model_dir,asText=True)

    graph.plot_weights(autoencoder,outdir = model_dir)
    graph.plot_weights(encoder,outdir = model_dir)
    graph.plot_weights(decoder,outdir = model_dir)
   
tf.saved_model.save(encoder, os.path.join(model_dir, 'best-encoder'))
tf.saved_model.save(decoder, os.path.join(model_dir, 'best-decoder'))
save_models(cae,args.mname,isQK = True)


