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

    
remap_8x8 = [ 4, 12, 20, 28,  5, 13, 21, 29,  6, 14, 22, 30,  7, 15, 23, 31, 
              24, 25, 26, 27, 16, 17, 18, 19,  8,  9, 10, 11,  0,  1,  2,  3, 
              59, 51, 43, 35, 58, 50, 42, 34, 57, 49, 41, 33, 56, 48, 40, 32]
def mean_mse_loss(y_true, y_pred):
    
    max_values = tf.reduce_max(y_true[:,], axis=1)
    
    y_true = tf.gather(K.reshape(y_true,(-1,64)),remap_8x8,axis=-1)
    y_pred = tf.gather(K.reshape(y_pred,(-1,64)),remap_8x8,axis=-1)
    # Calculate the squared difference between predicted and target values
    squared_diff = tf.square(y_pred - y_true)

    # Calculate the MSE per row (reduce_mean along axis=1)
    mse_per_row = tf.reduce_mean(squared_diff, axis=1)
    weighted_mse_per_row = mse_per_row * max_values
    

    # Take the mean of the MSE values to get the overall MSE loss
    mean_mse_loss = tf.reduce_mean(weighted_mse_per_row)
    return mean_mse_loss

args = p.parse_args()
model_dir = args.opath
if not os.path.exists(model_dir):
    os.system("mkdir -p "+model_dir)

batch = args.batchsize

n_kernels = 8
n_encoded=16
conv_weightBits  = 6 
conv_biasBits  = 6 
dense_weightBits  = 6 
dense_biasBits  = 6 
encodedBits = 9
CNN_kernel_size = 3
padding = tf.constant([[0,0],[0, 1], [0, 1], [0, 0]])


input_enc = Input(batch_shape=(batch,8,8, 1))
# sum_input quantization is done in the dataloading step for simplicity
# sum_input = Input(batch_shape=(batch,1))
# eta = Input(batch_shape =(batch,1))


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

# x = concatenate([x,sum_input,eta],axis=1)

latent = x

input_dec = Input(batch_shape=(batch,16))
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

cae = Model(
    inputs=[input_enc],
    outputs=decoder([encoder([input_enc])]),
    name="cae"
)

if args.loss == 'mse':
    loss=mean_mse_loss
elif args.loss == 'tele':
    loss = telescopeMSE8x8
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
        elif  type(v)==type(telescopeMSE8x8):
            jsonpams[k] =str(v) 
        else:
            jsonpams[k] = v 
    return jsonpams
def get_normalize(data,rescaleInputToMax=False, sumlog2=True):
    maxes =[]
    sums =[]
    sums_log2=[]
    for i in range(len(data)):
        maxes.append(data[i].max())
        sums.append(data[i].sum())
        sums_log2.append( 2**(np.floor(np.log2(data[i].sum()))) )
        if sumlog2:
            data[i] = 1.*data[i]/(sums_log2[-1] if sums_log2[-1] else 1.)
        elif rescaleInputToMax:
            data[i] = 1.*data[i]/(data[i].max() if data[i].max() else 1.)
        else:
            data[i] = 1.*data[i]/(data[i].sum() if data[i].sum() else 1.)
    if sumlog2:
        return  data,np.array(maxes),np.array(sums_log2)
    else:
        return data,np.array(maxes),np.array(sums)
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
    


def load_matching_state_dict(model, state_dict_path):
    state_dict = tf.compat.v1.train.load_checkpoint(state_dict_path)
    model_variables = model.trainable_variables
    filtered_state_dict = {}
    for var in model_variables:
        var_name = var.name.split(':')[0]
        if var_name in state_dict:
            filtered_state_dict[var_name] = state_dict[var_name]
    tf.compat.v1.train.init_from_checkpoint(state_dict_path, filtered_state_dict)

# Loading Model
if args.continue_training:
    
    cae.load_weights(args.mpath)
    start_epoch = int(args.mpath.split("/")[-1].split(".")[-2].split("-")[-1]) + 1
    print(f"Continuing training from epoch {start_epoch}...")
elif args.mpath:
    cae.load_weights(args.mpath)
#     load_matching_state_dict(cae, args.mpath)
    print('loaded model')

    

'''

//printf("msb(modSum_) = %u\n", msb(modSum_));
shift = msb(modSum_) - bitsPerInput_ + 1;
//printf("shift = %d\n", shift);

for(unsigned u=0; u<cellUVSize_; u++){
    for(unsigned v=0; v<cellUVSize_; v++){
        size_t CALQ = CALQs_[u][v];
        if(bitShiftNormalize_){
            if(shift > 0){
                CALQ = CALQ >> shift; //truncate
            } else {
                CALQ = CALQ << -shift;
            }
        } else {
            CALQ = size_t(getInputNorm() * CALQ / double(modSum_));//round
        }
        inputs_[u][v] = CALQ;
    }
'''


def load_data(nfiles,batchsize, normalize = True):
    ecr = np.vectorize(encode)
    data_list = []

    for i in range(nfiles):
        if i == 0:
            dt = pd.read_csv('../ECON_AE_Development/AE_Data/1.csv').values
        else:
            dt_i = pd.read_csv(f'../ECON_AE_Development/AE_Data/{i+1}.csv').values
            dt = np.vstack([dt, dt_i])

        data_list.append(dt)

    data_tensor = tf.convert_to_tensor(np.concatenate(data_list), dtype=tf.float32)
    data_tensor = data_tensor[0:int(300000)]
    train_size = int(0.9 * len(data_tensor))
    test_size = len(data_tensor) - train_size

    # Split the data into training and test sets
    train_data, test_data = tf.split(data_tensor, [train_size, test_size], axis=0)

    # Extract specific tensors
#     if normalize:
#         train_sum_calcq = tf.expand_dims(tf.sqrt(tf.reduce_sum(train_data[:, 0:48]**2, axis=1)), axis=1)
#         train_data = tf.boolean_mask(train_data,tf.squeeze(train_sum_calcq,axis=-1) != 0.0)
#         train_sum_calcq = tf.boolean_mask(train_sum_calcq,tf.squeeze(train_sum_calcq,axis=-1) != 0)
#         train_wafers = expand_tensor(train_data[:, 0:48]/train_sum_calcq)
#     else:
#         train_sum_calcq = tf.expand_dims(tf.reduce_sum(train_data[:, 0:48], axis=1), axis=1)
#         train_wafers = expand_tensor(train_data[:, 0:48])
    

#     if normalize:
#         test_sum_calcq = tf.expand_dims(tf.sqrt(tf.reduce_sum(test_data[:, 0:48]**2, axis=1)), axis=1)
#         test_data = tf.boolean_mask(test_data,tf.squeeze(test_sum_calcq,axis=-1) != 0.0)
#         test_sum_calcq = tf.boolean_mask(test_sum_calcq,tf.squeeze(test_sum_calcq,axis=-1) != 0)
#         test_wafers = expand_tensor(test_data[:, 0:48]/test_sum_calcq)
#     else:
#         test_sum_calcq = tf.expand_dims(tf.reduce_sum(test_data[:, 0:48], axis=1), axis=1)
#         test_wafers = expand_tensor(test_data[:, 0:48])
    if normalize:
        train_sum_calcq = tf.expand_dims(tf.reduce_sum(train_data[:, 0:48], axis=1), axis=1)
        train_data = tf.boolean_mask(train_data,tf.squeeze(train_sum_calcq,axis=-1) != 0.0)
        train_sum_calcq = tf.boolean_mask(train_sum_calcq,tf.squeeze(train_sum_calcq,axis=-1) != 0)
        train_wafers = get_normalize(train_data[:, 0:48].numpy())[0]
        train_wafers = expand_tensor(tf.convert_to_tensor(train_wafers, dtype=tf.float32))
    else:
        train_sum_calcq = tf.expand_dims(tf.reduce_sum(train_data[:, 0:48], axis=1), axis=1)
        train_wafers = expand_tensor(train_data[:, 0:48])
    

    if normalize:
        test_sum_calcq = tf.expand_dims(tf.reduce_sum(test_data[:, 0:48], axis=1), axis=1)
        test_data = tf.boolean_mask(test_data,tf.squeeze(test_sum_calcq,axis=-1) != 0.0)
        test_sum_calcq = tf.boolean_mask(test_sum_calcq,tf.squeeze(test_sum_calcq,axis=-1) != 0)
        test_wafers = get_normalize(test_data[:, 0:48].numpy())[0]
        test_wafers = expand_tensor(tf.convert_to_tensor(test_wafers, dtype=tf.float32))
        
    else:
        test_sum_calcq = tf.expand_dims(tf.reduce_sum(test_data[:, 0:48], axis=1), axis=1)
        test_wafers = expand_tensor(test_data[:, 0:48])
    
    ''' 
    
    Quantize Sum CalcQ with 5 exp bits and 4 mant bits
    
    Taking the log here is effectivel taking the log on readout. It makes no change to the quantization,
    just a convenient setup for keras.
    
    '''
   
    train_sum_calcq = train_sum_calcq.numpy().astype(int)
    train_sum_calcq = ecr(train_sum_calcq, dropBits=0, expBits=5, mantBits=4, roundBits=False, asInt=True)
    train_sum_calcq = tf.math.log(tf.constant(train_sum_calcq,dtype = tf.float64))
   
    
    test_sum_calcq = test_sum_calcq.numpy().astype(int)
    test_sum_calcq = ecr(test_sum_calcq, dropBits=0, expBits=5, mantBits=4, roundBits=False, asInt=True)
    test_sum_calcq = tf.math.log(tf.constant(test_sum_calcq,dtype = tf.float64))
    
    train_eta = tf.expand_dims(train_data[:, -2], axis=1)
    test_eta = tf.expand_dims(test_data[:, -2], axis=1)
    
    # Create data loaders for training and test data
    train_dataset = tf.data.Dataset.from_tensor_slices((train_wafers, train_sum_calcq, train_eta))
    train_loader = train_dataset.batch(batchsize).shuffle(buffer_size=train_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_wafers, test_sum_calcq, test_eta))
    test_loader = test_dataset.batch(batchsize).shuffle(buffer_size=test_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_loader, test_loader



def expand_tensor(input_tensor):
    arrange = np.array([28,29,30,31,0,4,8,12,
                         24,25,26,27,1,5,9,13,
                         20,21,22,23,2,6,10,14,
                         16,17,18,19,3,7,11,15,
                         47,43,39,35,35,34,33,32,
                         46,42,38,34,39,38,37,36,
                         45,41,37,33,43,42,41,40,
                             44,40,36,32,47,46,45,44])
    arrMask = np.array([1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,
                        1,1,1,1,0,0,0,0,
                        1,1,1,1,0,0,0,0,
                        1,1,1,1,0,0,0,0,
                        1,1,1,1,0,0,0,0,])
   
    inputdata = tf.reshape(tf.gather(input_tensor, arrange,axis =1), (input_tensor.shape[0],8, 8, 1))


#     paddings = [(0, 0), (0, 1), (0, 1), (0, 0)]
#     padded_tensor = tf.pad(inputdata, paddings, mode='CONSTANT', constant_values=0)

#     return padded_tensor
    return inputdata

print('Loading Data')
if args.batchsize != 1:
    train_loader, test_loader = load_data(args.num_files,batch)
print('Data Loaded')


best_val_loss = 1e9

all_train_loss = []
all_val_loss = []


if args.continue_training:
    cut_path = args.mpath.rsplit('/', 2)[0] + '/'
    loss_dict = {'train_loss': pd.read_csv(os.path.join(cut_path,'loss.csv'))['train_loss'].tolist(), 
 'val_loss': pd.read_csv(os.path.join(cut_path,'loss.csv'))['val_loss'].tolist()}
else:
    start_epoch = 1
    loss_dict = {'train_loss': [], 'val_loss': []}
  


for epoch in range(start_epoch, args.nepochs):
    if epoch == 20:
        if args.pretrain_model:
            print('Beginnning Fine Tuning')
            if args.optim == 'adam':
                opt = tf.keras.optimizers.Adam(learning_rate = args.lr,weight_decay = 0.000025)
            elif args.optim == 'lion':
                opt = tf.keras.optimizers.Lion(learning_rate = args.lr,weight_decay = 0.00025)
            cae.compile(optimizer=opt, loss=telescopeMSE8x8)
            
            cae.load_weights(model_dir+'/best-epoch.tf')
            print('Loaded Best Pretrained Model')
            
            
    
    total_loss_train = 0
    
    for wafers, sum_calcq, eta in train_loader:
        
#         if wafers.shape[0] != batch:
#             break
        loss = cae.train_on_batch([wafers], wafers)
        total_loss_train = total_loss_train + loss
        
    total_loss_val = 0 
    for wafers, sum_calcq, eta in test_loader:

#         if wafers.shape[0] != batch:
#             break
            
        loss = cae.test_on_batch([wafers], wafers)
        
        total_loss_val = total_loss_val+loss
        
        
    total_loss_train = total_loss_train/(len(train_loader))
    total_loss_val = total_loss_val/(len(test_loader))
    print('Epoch {:03d}, Loss: {:.8f}, ValLoss: {:.8f}'.format(
        epoch, total_loss_train,  total_loss_val))

    
    
    loss_dict['train_loss'].append(total_loss_train)
    loss_dict['val_loss'].append(total_loss_val)
    
    df = pd.DataFrame.from_dict(loss_dict)

    
    df.to_csv("%s/" % model_dir + "/loss.csv")

    cae.save_weights(os.path.join(model_dir, f'epoch-{epoch}.tf'))
    if total_loss_val < best_val_loss:
        print('New Best Model')
        best_val_loss = total_loss_val
        cae.save_weights(os.path.join(model_dir, 'best-epoch.tf'.format(epoch)))
        
# tf.saved_model.save(encoder, os.path.join(model_dir, 'best-encoder'))
# tf.saved_model.save(decoder, os.path.join(model_dir, 'best-decoder'))
save_models(cae,args.mname,isQK = True)

