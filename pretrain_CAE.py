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
    ('--num_files', p.INT),('--optim', p.STR),('--eLinks', p.INT),('--emd_pth', p.STR),('--sim_e_cut', p.STORE_TRUE),('--e_cut', p.STORE_TRUE),('--biased', p.STORE_TRUE),('--b_percent', {'type': float}),('--flat_eta', p.STORE_TRUE),('--flat_sum_CALQ', p.STORE_TRUE)
    
    
    
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

def save_models(autoencoder, name, isQK=False):

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
    graph.write_frozen_graph_enc(encoder,'encoder_'+name+'.pb',logdir = model_dir)
    graph.write_frozen_graph_enc(encoder,'encoder_'+name+'.pb.ascii',logdir = model_dir,asText=True)
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

def load_data(normalize = True,eLinks = -1):
    from files import get_rootfiles
    from coffea.nanoevents import NanoEventsFactory
    import awkward as ak
    import numpy as np
    ecr = np.vectorize(encode)
    data_list = []
    

    hostid = 'cmseos.fnal.gov'
    basepath = '/store/group/lpcpfnano/srothman/Nov08_2023_ECON_trainingdata'
    tree = 'FloatingpointThreshold0DummyHistomaxDummynTuple/HGCalTriggerNtuple'
    
    files = get_rootfiles(hostid, basepath)[0:50]
    

    #loop over all the files
    for i,file in enumerate(files):
        x = NanoEventsFactory.from_root(file, treepath=tree).events()
        
        min_pt = 0 # replace with your minimum value
        max_pt = 100000  # replace with your maximum value
        gen_pt = ak.to_pandas(x.gen.pt).groupby(level=0).mean()
        mask = (gen_pt['values'] >= min_pt) & (gen_pt['values'] <= max_pt)
        layers = ak.to_pandas(x.wafer.layer)
        layers = layers.loc[layers.index.get_level_values('entry').isin(mask)]

        eta = ak.to_pandas(x.wafer.eta)
        eta = eta.loc[eta.index.get_level_values('entry').isin(mask)]

        waferv = ak.to_pandas(x.wafer.waferv)
        waferv = waferv.loc[waferv.index.get_level_values('entry').isin(mask)]

        waferu = ak.to_pandas(x.wafer.waferu)
        waferu = waferu.loc[waferu.index.get_level_values('entry').isin(mask)]

        wafertype = ak.to_pandas(x.wafer.wafertype)
        wafertype = wafertype.loc[wafertype.index.get_level_values('entry').isin(mask)]

        sumCALQ = ak.to_pandas(x.wafer['CALQ0'])
        sumCALQ = sumCALQ.loc[sumCALQ.index.get_level_values('entry').isin(mask)]
        
        
        wafer_sim_energy = ak.to_pandas(x.wafer.simenergy)
        wafer_sim_energy = wafer_sim_energy.loc[wafer_sim_energy.index.get_level_values('entry').isin(mask)]
        
        wafer_energy = ak.to_pandas(x.wafer.energy)
        wafer_energy = wafer_energy.loc[wafer_energy.index.get_level_values('entry').isin(mask)]
        
        
        layers = np.squeeze(layers.to_numpy())
        eta = np.squeeze(eta.to_numpy())
        waferv = np.squeeze(waferv.to_numpy())
        waferu = np.squeeze(waferu.to_numpy())
        temp = np.squeeze(wafertype.to_numpy())
        wafertype = np.zeros((temp.size, temp.max() + 1))
        wafertype[np.arange(temp.size), temp] = 1
        sumCALQ = np.squeeze(sumCALQ.to_numpy())
        wafer_sim_energy = np.squeeze(wafer_sim_energy.to_numpy())
        wafer_energy = np.squeeze(wafer_energy.to_numpy())
        
        

        for i in range(1,64):
            cur = ak.to_pandas(x.wafer[f'CALQ{int(i)}'])
            cur = cur.loc[cur.index.get_level_values('entry').isin(mask)]
            cur = np.squeeze(cur.to_numpy())
            sumCALQ = sumCALQ + cur
            
        sumCALQ = np.log(sumCALQ+1)
        
        inputs = []
        for i in range(64):
            cur = ak.to_pandas(x.wafer['AEin%d'%i])
            cur = cur.loc[cur.index.get_level_values('entry').isin(mask)]
            cur = np.squeeze(cur.to_numpy())
            inputs.append(cur) 
       
        inputs = np.stack(inputs, axis=-1) #stack all 64 inputs
        inputs = np.reshape(inputs, (-1, 8, 8))
        select_eLinks = {5 : (layers<=11) & (layers>=5) ,
                 4 : (layers==7) | (layers==11),
                 3 : (layers==13),
                 2 : (layers<7) | (layers>13),
                 -1 : (layers>0)}
        inputs = inputs[select_eLinks[eLinks]]
        l =layers[select_eLinks[eLinks]]
        eta = eta[select_eLinks[eLinks]]
        waferv = waferv[select_eLinks[eLinks]]
        waferu = waferu[select_eLinks[eLinks]]
        wafertype = wafertype[select_eLinks[eLinks]]
        sumCALQ = sumCALQ[select_eLinks[eLinks]]
        wafer_sim_energy = wafer_sim_energy[select_eLinks[eLinks]]
        wafer_energy = wafer_energy[select_eLinks[eLinks]]
        data_list.append([inputs,eta,waferv,waferu,wafertype,sumCALQ,l,wafer_sim_energy])


    inputs_list = []
    eta_list = []
    waferv_list = []
    waferu_list = []
    wafertype_list = []
    sumCALQ_list = []
    layer_list = []
    wafer_sim_e = []

    for item in data_list:
        inputs, eta, waferv, waferu, wafertype, sumCALQ,layers,wafer_sim_energy = item
        inputs_list.append(inputs)
        eta_list.append(eta)
        waferv_list.append(waferv)
        waferu_list.append(waferu)
        wafertype_list.append(wafertype)
        sumCALQ_list.append(sumCALQ)
        layer_list.append(layers)
        wafer_sim_e.append(wafer_sim_energy)

    concatenated_inputs = np.expand_dims(np.concatenate(inputs_list),axis = -1)
    concatenated_eta = np.expand_dims(np.concatenate(eta_list),axis = -1)
    concatenated_waferv = np.expand_dims(np.concatenate(waferv_list),axis = -1)
    concatenated_waferu = np.expand_dims(np.concatenate(waferu_list),axis = -1)
    concatenated_wafertype = np.concatenate(wafertype_list)
    concatenated_sumCALQ = np.expand_dims(np.concatenate(sumCALQ_list),axis = -1)
    concatenated_layers = np.expand_dims(np.concatenate(layer_list),axis = -1)
    concatenated_sim_e = np.expand_dims(np.concatenate(wafer_sim_e),axis = -1)
    
    concatenated_cond = np.hstack([concatenated_eta,concatenated_waferv,concatenated_waferu, concatenated_wafertype, concatenated_sumCALQ,concatenated_layers,concatenated_sim_e])
    
    
    signals_mask = tf.not_equal(cond[:, -1], 0)
    signals = inputs[signals_mask]
    signals_cond = cond[signals_mask]
    def signal_noise_pairs_generator():
        # Loop through all signal entries
        for signal, signal_cond in zip(signals, signals_cond):
            # Conditions for matching: same values at indices 1 and 2
            matches = tf.logical_and(
                tf.equal(cond[:, 1], signal_cond[1]),
                tf.equal(cond[:, 2], signal_cond[2])
            )
            # Ensure we also select only noise entries
            matches = tf.logical_and(matches, tf.equal(cond[:, 0], 0))

            # Find matching noise inputs
            matching_noise_inputs = inputs[matches]

            # Add the first matching noise input to the signal to create an augmented signal
            # Note: If no match is found, return the original signal to handle empty cases
            if tf.shape(matching_noise_inputs)[0] > 0:
                augmented_signal = signal + matching_noise_inputs[0]
            else:
                augmented_signal = signal

            yield signal, augmented_signal

    output_types = (tf.float32, tf.float32)  # Assuming the inputs are of type float32
    output_shapes = ((8, 8), (8, 8))
    all_dataset = tf.data.Dataset.from_generator(
        signal_noise_pairs_generator,
        output_types=output_types,
        output_shapes=output_shapes
    ).take(1000000)


    total_size = len(all_dataset)  # Replace with your dataset's total size
    print('total size: ',total_size)
    
    # Define your splitting ratio
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size

    # Create the training dataset
    train_dataset = all_dataset.take(train_size)

    # Create the test dataset
    test_dataset = all_dataset.skip(train_size).take(test_size)

    train_loader = train_dataset.batch(batchsize).shuffle(buffer_size=train_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    test_loader = test_dataset.batch(batchsize).shuffle(buffer_size=test_size).prefetch(buffer_size=tf.data.AUTOTUNE)


    return train_loader, test_loader
    
def get_encoder(eLinks):
    bitsPerOutputLink = [0, 1, 3, 5, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
    
    print(f'Training Model with {eLinks} eLinks')
    model_dir = os.path.join(args.opath, f'model_{eLinks}_eLinks')
    if not os.path.exists(model_dir):
        os.system("mkdir -p " + model_dir)
    bitsPerOutput = bitsPerOutputLink[eLinks]
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


    input_enc = Input(batch_shape=(batch,8,8, 1), name = 'Wafer')
    # sum_input quantization is done in the dataloading step for simplicity
    
    cond = Input(batch_shape=(batch, 8), name = 'Cond')


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


    x = Flatten()(x)

    x = QDense(n_encoded, 
               kernel_quantizer=quantized_bits(bits=dense_weightBits,integer=0,keep_negative=1,alpha=1),
               bias_quantizer=quantized_bits(bits=dense_biasBits,integer=0,keep_negative=1,alpha=1),
               name="dense")(x)

    # Quantizing latent space, 9 bit quantization, 1 bit for integer
    x = QActivation(qkeras.quantized_bits(bits = 9, integer = 1),name = 'latent_quantization')(x)
    latent = x
    if bitsPerOutput > 0 and maxBitsPerOutput > 0:
        latent = tf.minimum(tf.math.floor(latent *  outputMaxIntSize) /  outputMaxIntSize, outputSaturationValue)
    
    
    encoder = keras.Model([input_enc,latent], latent, name="encoder")
    return encoder
    
args = p.parse_args()
model_dir = args.opath
if not os.path.exists(model_dir):
    os.system("mkdir -p "+model_dir)

class ContrastiveModel(keras.Model):
    def __init__(self,eLinks):
        super().__init__()

        self.temperature = temperature
        self.encoder = get_encoder(eLinks)

        self.encoder.summary()

    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer

        # self.contrastive_loss will be defined as a method
        self.probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name="c_acc"
        )
        self.probe_loss_tracker = keras.metrics.Mean(name="p_loss")
        self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy(name="p_acc")

    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker,
            self.contrastive_accuracy,
            self.probe_loss_tracker,
            self.probe_accuracy,
        ]

    def contrastive_loss(self, projections_1, projections_2):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = ops.normalize(projections_1, axis=1)
        projections_2 = ops.normalize(projections_2, axis=1)
        similarities = (
            ops.matmul(projections_1, ops.transpose(projections_2)) / self.temperature
        )

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = ops.shape(projections_1)[0]
        contrastive_labels = ops.arange(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(
            contrastive_labels, ops.transpose(similarities)
        )

        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, ops.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2

    def train_step(self, data):
        signal, augmented = data

        with tf.GradientTape() as tape:
            features_1 = self.encoder(signal, training=True)
            features_2 = self.encoder(augmented, training=True)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.contrastive_loss_tracker.update_state(contrastive_loss)

        # Labels are only used in evalutation for an on-the-fly logistic regression
        preprocessed_images = self.classification_augmenter(
            labeled_images, training=True
        )
        with tf.GradientTape() as tape:
            # the encoder is used in inference mode here to avoid regularization
            # and updating the batch normalization paramers if they are used
            features = self.encoder(preprocessed_images, training=False)
            class_logits = self.linear_probe(features, training=True)
            probe_loss = self.probe_loss(labels, class_logits)
        gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        self.probe_optimizer.apply_gradients(
            zip(gradients, self.linear_probe.trainable_weights)
        )
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        labeled_images, labels = data

        # For testing the components are used with a training=False flag
        preprocessed_images = self.classification_augmenter(
            labeled_images, training=False
        )
        features = self.encoder(preprocessed_images, training=False)
        class_logits = self.linear_probe(features, training=False)
        probe_loss = self.probe_loss(labels, class_logits)
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)

        # Only the probe metrics are logged at test time
        return {m.name: m.result() for m in self.metrics[2:]}


# Contrastive pretraining
pretraining_model = ContrastiveModel()
pretraining_model.compile(
    contrastive_optimizer=keras.optimizers.Adam(),
    probe_optimizer=keras.optimizers.Adam(),
)

pretraining_history = pretraining_model.fit(
    train_dataset, epochs=num_epochs, validation_data=test_dataset
)
print(
    "Maximal validation accuracy: {:.2f}%".format(
        max(pretraining_history.history["val_p_acc"]) * 100
    )
)
    
# Loop through each number of eLinks
for eLinks in [2,3,4,5]:
     

    if args.optim == 'adam':
        print('Using ADAM Optimizer')
        opt = tf.keras.optimizers.Adam(learning_rate = args.lr,weight_decay = 0.000025)
    elif args.optim == 'lion':
        print('Using Lion Optimizer')
        opt = tf.keras.optimizers.Lion(learning_rate = args.lr,weight_decay = 0.00025)

    encoder.compile(optimizer=opt, loss=loss)
    encoder.summary()



    # Loading Model
    if args.continue_training:

        encoder.load_weights(args.mpath)
        start_epoch = int(args.mpath.split("/")[-1].split(".")[-2].split("-")[-1]) + 1
        print(f"Continuing training from epoch {start_epoch}...")
    elif args.mpath:
        encoder.load_weights(args.mpath)
        print('loaded model')





    print('Loading Data')
    train_loader, test_loader = load_data(args.num_files,batch,eLinks =eLinks)
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

        for wafers, cond in train_loader:

            loss = encoder.train_on_batch([wafers,cond], wafers)
            total_loss_train = total_loss_train + loss

        total_loss_val = 0 
        for wafers, cond in test_loader:


            loss = encoder.test_on_batch([wafers, cond], wafers)

            total_loss_val = total_loss_val+loss


        total_loss_train = total_loss_train#/(len(train_loader))
        total_loss_val = total_loss_val#/(len(test_loader))
        if epoch % 25 == 0:
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
        plt.legend()
        plt.grid(True)

        # Saving the plot in the same directory as the loss CSV
        plot_path = f"{model_dir}/loss_plot.png"
        plt.savefig(plot_path)

        if total_loss_val < best_val_loss:
            if epoch % 25 == 0:
                print('New Best Model')
            best_val_loss = total_loss_val
            encoder.save_weights(os.path.join(model_dir, 'best-epoch.tf'.format(epoch)))
            encoder.save_weights(os.path.join(model_dir, 'best-encoder-epoch.tf'.format(epoch)))
            decoder.save_weights(os.path.join(model_dir, 'best-decoder-epoch.tf'.format(epoch)))
    save_models(encoder,args.mname,isQK = True)

