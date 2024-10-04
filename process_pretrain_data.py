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



p = ArgumentParser()
p.add_args(
    ('--opath', p.STR),
    ('--num_files', p.INT),
    ('--model_per_eLink',  p.STORE_TRUE),
    ('--model_per_bit_config',  p.STORE_TRUE),
    ('--biased', {'type': float}),
    ('--alloc_geom', p.STR))
    
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
def custom_resample(wafers,c,simE):
    
    label = (simE[:,0] != 0).astype(int)
    n = len(label)
    print(Counter(label))
    indices = np.expand_dims(np.arange(n),axis = -1)
    # 10x upsample signal
    over = RandomOverSampler(sampling_strategy=0.1)
    indices_p, label_p = over.fit_resample(indices, label)
    # downsample until 1:2::pilup:signal
    signal_percent = 1-args.biased
    ratio = args.biased / signal_percent
    if ratio > 1:
        ratio = 1 / ratio
        under = RandomUnderSampler(sampling_strategy=ratio)
        indices_p, label_p = under.fit_resample(indices_p, label_p)
    else:
        under = RandomUnderSampler(sampling_strategy=ratio)
        indices_p, label_p = under.fit_resample(indices_p, label_p)
    print(Counter(label_p))
    wafers_p = wafers[indices_p[:,0]]
    c_p = c[indices_p[:,0]]
    
    return wafers_p, c_p

def get_old_mask(eLink, df):
    # Initialize a mask with all False values, with the same index as the DataFrame
    mask = pd.Series([False] * len(df), index=df.index)
    
#     for eLink in eLinks:
    if eLink == 5:
        mask = mask | ((df['layer'] <= 11) & (df['layer'] >= 5))
    elif eLink == 4:
        mask = mask | ((df['layer'] == 7) | (df['layer'] == 11))
    elif eLink == 3:
        mask = mask | (df['layer'] == 13)
    elif eLink == 2:
        mask = mask | ((df['layer'] < 7) | (df['layer'] > 13))
    elif eLink == -1:
        mask = mask | (df['layer'] > 0)
    
    return mask
    
def save_data(nfiles,model_info = -1, normalize = True):
    from files import get_rootfiles
    from coffea.nanoevents import NanoEventsFactory
    import awkward as ak
    import numpy as np
    ecr = np.vectorize(encode)
    data_list = []
    simE_list = []
    if args.model_per_eLink:
        eLinks = model_info
    elif args.model_per_bit_config:
        bitsPerOutput = model_info

    # Paths to Simon's dataset
    hostid = 'cmseos.fnal.gov'
    basepath = '/store/group/lpcpfnano/srothman/Nov08_2023_ECON_trainingdata'
    tree = 'FloatingpointThreshold0DummyHistomaxDummynTuple/HGCalTriggerNtuple'

    files = get_rootfiles(hostid, basepath)[0:nfiles]

    os.environ["XRD_REQUESTTIMEOUT"] = "120"
    os.environ["XRD_REDIRECTTIMEOUT"] = "120"
    #loop over all the files
    num_saved_files = 0
    for i,file in enumerate(files):
        print('=====================')
        print(i)
        print('=====================')
        
        x = NanoEventsFactory.from_root(file, treepath=tree).events()

        min_pt = -1  # replace with your minimum value
        max_pt = 10e10  # replace with your maximum value
        gen_pt = ak.to_pandas(x.gen.pt).groupby(level=0).mean()
        mask = (gen_pt['values'] >= min_pt) & (gen_pt['values'] <= max_pt)
        
        
        layer = ak.to_pandas(x.wafer.layer)
        eta = ak.to_pandas(x.wafer.eta)
        v = ak.to_pandas(x.wafer.waferv)
        u = ak.to_pandas(x.wafer.waferu)
        wafertype = ak.to_pandas(x.wafer.wafertype)
        wafer_sim_energy = ak.to_pandas(x.wafer.simenergy)
        wafer_energy = ak.to_pandas(x.wafer.energy)
        
        # Combine all DataFrames into a single DataFrame
        data_dict = {
            'eta': eta.values.flatten(),
            'v': v.values.flatten(),
            'u': u.values.flatten(),
            'wafertype': wafertype.values.flatten(),
            'wafer_sim_energy': wafer_sim_energy.values.flatten(),
            'wafer_energy': wafer_energy.values.flatten(),
            'layer': layer.values.flatten()
        }

        # Add additional features AEin1 to AEin63 to the data dictionary
        key = 'AEin0'
        data_dict[key] = ak.to_pandas(x.wafer[key]).values.flatten()
        for j in range(1, 64):
            key = f'AEin{j}'
            data_dict[key] = ak.to_pandas(x.wafer[key]).values.flatten()
            key = f'CALQ{int(j)}'
            data_dict[key] = ak.to_pandas(x.wafer[key]).values.flatten()
            
        
        # Combine all data into a single DataFrame
        combined_df = pd.DataFrame(data_dict, index=eta.index)
        calq_columns = [f'CALQ{i}' for i in range(1,64)]
        combined_df['sumCALQ'] = combined_df[calq_columns].sum(axis=1)
        
        if args.alloc_geom == 'new':
            if args.model_per_eLink:
                filtered_key_df = key_df[key_df['trigLinks'] == float(eLinks)]
                filtered_df = pd.merge(combined_df, filtered_key_df[['u', 'v', 'layer']], on=['u', 'v', 'layer'], how='inner')
            elif args.model_per_bit_config:
                eLinks_with_bit_alloc = [index for index, value in enumerate(bitsPerOutputLink) if value == bitsPerOutput]
                eLinks_with_bit_alloc = [float(b) for b in eLinks_with_bit_alloc if b < 12]
                filtered_key_df = key_df[key_df['trigLinks'].isin(eLinks_with_bit_alloc)]
                filtered_df = pd.merge(combined_df, filtered_key_df[['u', 'v', 'layer']], on=['u', 'v', 'layer'], how='inner')
        
        elif args.alloc_geom =='old':
            if args.model_per_eLink:
                mask = get_old_mask(eLinks, combined_df)
                filtered_df = combined_df[mask]
            elif args.model_per_bit_config:
                eLinks_with_bit_alloc = [index for index, value in enumerate(bitsPerOutputLink) if value == bitsPerOutput]
                eLinks_with_bit_alloc = [b for b in eLinks_with_bit_alloc if b < 6]
                
                mask = get_old_mask(eLinks_with_bit_alloc, combined_df)
#                 print(combined_df)
                filtered_df = combined_df[mask]
                filtered_df = filtered_df.copy()
#                 print(filtered_df)
                
                
        print('Size after eLink filtering')
        print(len(filtered_df))
        
        # Process the filtered DataFrame
        filtered_df.loc[:,'eta'] = filtered_df['eta'] / 3.1
        filtered_df.loc[:,'v'] = filtered_df['v'] / 12
        filtered_df.loc[:,'u'] = filtered_df['u'] / 12
        filtered_df.loc[:,'layer'] = (filtered_df['layer']-1) / 46

        # Convert wafertype to one-hot encoding
        temp = filtered_df['wafertype'].astype(int).to_numpy()
        wafertype_one_hot = np.zeros((temp.size, 3))
        wafertype_one_hot[np.arange(temp.size), temp] = 1

        # Assign the processed columns back to the DataFrame
        filtered_df['wafertype'] = list(wafertype_one_hot)
        filtered_df['sumCALQ'] = np.squeeze(filtered_df['sumCALQ'].to_numpy())
        filtered_df['wafer_sim_energy'] = np.squeeze(filtered_df['wafer_sim_energy'].to_numpy())
        filtered_df['wafer_energy'] = np.squeeze(filtered_df['wafer_energy'].to_numpy())
        filtered_df['layer'] = np.squeeze(filtered_df['layer'].to_numpy())
        

        inputs = []
        for j in range(64):
            cur = filtered_df['AEin%d'%j]
            cur = np.squeeze(cur.to_numpy())
            inputs.append(cur) 
        inputs = np.stack(inputs, axis=-1) #stack all 64 inputs
        inputs = np.reshape(inputs, (-1, 8, 8))
        
        
        layer = filtered_df['layer'].to_numpy()
        eta = filtered_df['eta'].to_numpy()
        v = filtered_df['v'].to_numpy()
        u = filtered_df['u'].to_numpy()
        wafertype = np.array(filtered_df['wafertype'].tolist())
        sumCALQ = filtered_df['sumCALQ'].to_numpy()
        sumCALQ = np.log(sumCALQ+1)
        wafer_sim_energy = filtered_df['wafer_sim_energy'].to_numpy()
        wafer_energy = filtered_df['wafer_energy'].to_numpy()
        data_list.append([inputs,eta,v,u,wafertype,sumCALQ,layer])
        simE_list.append(wafer_sim_energy)
        
        if (i % 5 == 0 and i !=0) or i == len(files)-1:
            inputs_list = []
            eta_list = []
            v_list = []
            u_list = []
            wafertype_list = []
            sumCALQ_list = []
            layer_list = []

            for item in data_list:
                inputs, eta, v, u, wafertype, sumCALQ,layer = item
                inputs_list.append(inputs)
                eta_list.append(eta)
                v_list.append(v)
                u_list.append(u)
                wafertype_list.append(wafertype)
                sumCALQ_list.append(sumCALQ)
                layer_list.append(layer)

            concatenated_inputs = np.expand_dims(np.concatenate(inputs_list),axis = -1)
            concatenated_eta = np.expand_dims(np.concatenate(eta_list),axis = -1)
            concatenated_v = np.expand_dims(np.concatenate(v_list),axis = -1)
            concatenated_u = np.expand_dims(np.concatenate(u_list),axis = -1)
            concatenated_wafertype = np.concatenate(wafertype_list)
            concatenated_sumCALQ = np.expand_dims(np.concatenate(sumCALQ_list),axis = -1)
            concatenated_layer = np.expand_dims(np.concatenate(layer_list),axis = -1)
            concatenated_simE = np.expand_dims(np.concatenate(simE_list),axis = -1)
            concatenated_cond = np.hstack([concatenated_eta,concatenated_v,concatenated_u, concatenated_wafertype, concatenated_sumCALQ,concatenated_layer])



            events = len(concatenated_cond)
            indices = np.random.permutation(events)
            # Calculate 80% of n
            num_selected = int(0.8 * events)

            # Select the first 80% of the indices
            train_indices = indices[:num_selected]
            test_indices = indices[num_selected:]
            wafer_train = concatenated_inputs[train_indices]
            wafer_test = concatenated_inputs[test_indices]

            simE_train = concatenated_simE[train_indices]
            simE_test = concatenated_simE[test_indices]

            cond_train = concatenated_cond[train_indices]
            cond_test = concatenated_cond[test_indices]
            if args.biased:
                wafer_train,cond_train = custom_resample(wafer_train,cond_train,simE_train)
                print(wafer_train.shape)
                wafer_test,cond_test = custom_resample(wafer_test,cond_test, simE_test)
            

            # Create the training dataset
            train_dataset = tf.data.Dataset.from_tensor_slices((wafer_train,cond_train)
            )

            # Create the test dataset
            test_dataset = tf.data.Dataset.from_tensor_slices((wafer_test,cond_test)
            )

            tf.data.experimental.save(train_dataset, os.path.join(model_dir,f'{num_saved_files}_train'))
            tf.data.experimental.save(test_dataset, os.path.join(model_dir,f'{num_saved_files}_test'))
            data_list.append([inputs,eta,v,u,wafertype,sumCALQ,layer])
            simE_list.append(wafer_sim_energy)
            
            data_list = []
            simE_list = []
            
            num_saved_files += 1
    

    
args = p.parse_args()

# Loop through each number of eLinks

if args.model_per_eLink:
    if args.alloc_geom == 'old':
        all_models = [2,3,4,5]
    elif args.alloc_geom =='new':
        all_models = [1,2,3,4,5,6,7,8,9,10,11]
elif args.model_per_bit_config:
    if args.alloc_geom == 'old':
        all_models = [3]#[3,5,7,9]
    elif args.alloc_geom =='new':
        all_models = [1,3,5,7,9]


bitsPerOutputLink = [0, 1, 3, 5, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]    

for m in all_models:
    if args.model_per_eLink:
        eLinks = m
        bitsPerOutput = bitsPerOutputLink[eLinks]
        model_dir = os.path.join(args.opath, f'data_{eLinks}_eLinks')
    elif args.model_per_bit_config:
        bitsPerOutput = m
        model_dir = os.path.join(args.opath, f'data_{bitsPerOutput}_bits')
    
    if not os.path.exists(model_dir):
        os.system("mkdir -p " + model_dir)
        
    save_data(args.num_files,model_info = m)
    
    

    

