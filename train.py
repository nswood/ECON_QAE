import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from qkeras import *
from keras.models import Model
from keras.layers import *
from telescope import *

import os
import sys
import utils
p = utils.ArgumentParser()
p.add_args(
    ('--mname', p.STR),
    ('--loss', p.STR), ('--nepochs', p.INT),
    ('--opath', p.STR),
    ('--mpath', p.STR),('--prepath', p.STR),('--continue_training', p.STORE_TRUE), ('--batchsize', p.INT),
    ('--lr', {'type': float}),
    ('--num_files', p.INT),
    
    
    
)

def mean_mse_loss(y_true, y_pred):
    # Calculate the squared difference between predicted and target values
    squared_diff = tf.square(y_pred - y_true)

    # Calculate the MSE per row (reduce_mean along axis=1)
    mse_per_row = tf.reduce_mean(squared_diff, axis=1)

    # Take the mean of the MSE values to get the overall MSE loss
    mean_mse_loss = tf.reduce_mean(mse_per_row)

    return mean_mse_loss

args = p.parse_args()
model_dir = args.opath
if not os.path.exists(model_dir):
    os.system("mkdir -p "+model_dir)

batch = args.batchsize

#Specs
n_kernels = 8
n_encoded=16
conv_weightBits  = 6 
conv_biasBits  = 6 
dense_weightBits  = 6 
dense_biasBits  = 6 
encodedBits = 9
CNN_kernel_size = 3

input_enc = Input(batch_shape=(batch,9,9, 1))
sum_input = Input(batch_shape=(batch,1))
eta = Input(batch_shape =(batch,1))
x = QConv2D(n_kernels,
            CNN_kernel_size, 
#             padding='same',
            strides=2,
            kernel_quantizer=quantized_bits(bits=conv_weightBits,integer=0,keep_negative=1,alpha=1),
            bias_quantizer=quantized_bits(bits=conv_biasBits,integer=0,keep_negative=1,alpha=1),
            name="conv2d")(input_enc)
x = QActivation("quantized_relu(bits=8,integer=1)", name="act")(x)
x = Flatten()(x)

# x = Concatenate(axis=1)([x[:,:80],x[:,96:112]]) 
x = QDense(n_encoded, 
           kernel_quantizer=quantized_bits(bits=dense_weightBits,integer=0,keep_negative=1,alpha=1),
           bias_quantizer=quantized_bits(bits=dense_biasBits,integer=0,keep_negative=1,alpha=1),
           name="dense")(x)

x = concatenate([x,sum_input,eta],axis=1)
latent = x


input_dec = Input(batch_shape=(batch,18))
y = Dense(128)(input_dec)
y = ReLU()(y)
y = Reshape((4, 4, 8))(y)
y = Conv2DTranspose(1, (3, 3), strides=(2, 2))(y)
y = ReLU()(y)
recon = y


encoder = keras.Model([input_enc,sum_input,eta], latent, name="encoder")
decoder = keras.Model([input_dec], recon, name="decoder")

cae = Model(
    inputs=[input_enc,sum_input,eta],
    outputs=decoder([encoder([input_enc,sum_input,eta])]),
    name="cae"
)

cae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-5), loss=mean_mse_loss)
cae.summary()

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
    checkpoint = tf.train.latest_checkpoint(args.mpath)
    model.load_weights(checkpoint)
    start_epoch = int(checkpoint.split("/")[-1].split("-")[-1]) + 1
    print(f"Continuing training from epoch {start_epoch}...")
elif args.mpath:
    load_matching_state_dict(model, args.mpath)
    print('loaded model')


def load_data(nfiles,batchsize):
    data_list = []

    for i in range(nfiles):
        if i == 0:
            dt = pd.read_csv('../ECON_AE_Development/AE_Data/1.csv').values
        else:
            dt_i = pd.read_csv(f'../ECON_AE_Development/AE_Data/{i+1}.csv').values
            dt = np.vstack([dt, dt_i])

        data_list.append(dt)

    data_tensor = tf.convert_to_tensor(np.concatenate(data_list), dtype=tf.float32)

    train_size = int(0.8 * len(data_tensor))
    test_size = len(data_tensor) - train_size

    # Split the data into training and test sets
    train_data, test_data = tf.split(data_tensor, [train_size, test_size], axis=0)

    # Extract specific tensors
    train_wafers = expand_tensor(train_data[:, 0:48])
    train_sum_calcq = tf.expand_dims(tf.reduce_sum(train_data[:, 0:48], axis=1), axis=1)
    train_eta = tf.expand_dims(train_data[:, -2], axis=1)

    test_wafers = expand_tensor(test_data[:, 0:48])
    test_sum_calcq = tf.expand_dims(tf.reduce_sum(test_data[:, 0:48], axis=1), axis=1)
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
#     inputdata *= tf.cast(tf.reshape(arrMask, (1, 8, 8)), dtype=inputdata.dtype)

    paddings = [(0, 0), (0, 1), (0, 1), (0, 0)]
    padded_tensor = tf.pad(inputdata, paddings, mode='CONSTANT', constant_values=0)

    return padded_tensor

print('Loading Data')
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
    total_loss_train = 0
    
    for wafers, sum_calcq, eta in test_loader:
        
        if wafers.shape[0] != batch:
            break
            
        loss = cae.train_on_batch([wafers, sum_calcq, eta], wafers)
        total_loss_train += loss
    total_loss_val = 0 
    
    for wafers, sum_calcq, eta in test_loader:

        if wafers.shape[0] != batch:
            break
            
        loss = cae.test_on_batch([wafers, sum_calcq, eta], wafers)
        total_loss_val += loss
        
        
    total_loss_train = total_loss_train/(len(train_loader)*batch)
    total_loss_val = total_loss_val/(len(test_loader)*batch)
    
    print('Epoch {:03d}, Loss: {:.8f}, ValLoss: {:.8f}'.format(
        epoch, total_loss_train,  total_loss_val))

    
    
    loss_dict['train_loss'].append(total_loss_train)
    loss_dict['val_loss'].append(total_loss_val)
    
    df = pd.DataFrame.from_dict(loss_dict)

    
    df.to_csv("%s/" % model_dir + "/loss.csv")

    cae.save_weights(os.path.join(model_dir, f'epoch-{epoch}.tf'))
    if total_loss_val < best_val_loss:
        best_val_loss = total_loss_val
        cae.save_weights(os.path.join(model_dir, 'best-epoch.tf'.format(epoch)))
