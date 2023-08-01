# ECON_QAE
Weight training of the autoencoder in QKeras using a conditional autoencoder architecture for the future implementation of the HGCAL. A conditional autoencoder is an optimal architecture for this limited-sized model for several reasons. 

It is widely known that the normalization of data is extremely beneficial to the training of models. However, normalization here disregards the underlying physics and throws away key information for future physical analysis. We can solve this problem through normalization of the data, to allow for optimal training, while conditioning on the sum of the input during decoding, thus preserving key physical information. 

Additionally, we have seen a strong eta dependence on readout characteristics. This stems from three different thicknesses for the silicon wafers which detect particles that are ordered based on eta. We can resolve this problem as well through conditioning on the known eta value of the silicon wafer during decoding. Thus we can effectively build the system geometry into our model.


## Encoder Architecture
The encoder is designed to be implemented on a chip so we have a strict limitation on the encoder size.
Model: "encoder"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_61 (InputLayer)          [(2000, 9, 9, 1)]    0           []                               
                                                                                                  
 conv2d (QConv2D)               (2000, 4, 4, 8)      80          ['input_61[0][0]']               
                                                                                                  
 act (QActivation)              (2000, 4, 4, 8)      0           ['conv2d[0][0]']                 
                                                                                                  
 flatten_15 (Flatten)           (2000, 128)          0           ['act[0][0]']                    
                                                                                                  
 dense (QDense)                 (2000, 16)           2064        ['flatten_15[0][0]']             
                                                                                                  
 input_62 (InputLayer)          [(2000, 1)]          0           []                               
                                                                                                  
 input_63 (InputLayer)          [(2000, 1)]          0           []                               
                                                                                                  
 concatenate_19 (Concatenate)   (2000, 18)           0           ['dense[0][0]',                  
                                                                  'input_62[0][0]',               
                                                                  'input_63[0][0]']               
                                                                                                  
==================================================================================================
Total params: 2,144
Trainable params: 2,144
Non-trainable params: 0
__________________________________________________________________________________________________


## Decoder Architecture
The decoder has no size limitations or quantization requirements. We utilize a larger and unquantized decoder to aid in finding the optimal encoder:
Model: "encoder"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_61 (InputLayer)          [(2000, 9, 9, 1)]    0           []                               
                                                                                                  
 conv2d (QConv2D)               (2000, 4, 4, 8)      80          ['input_61[0][0]']               
                                                                                                  
 act (QActivation)              (2000, 4, 4, 8)      0           ['conv2d[0][0]']                 
                                                                                                  
 flatten_15 (Flatten)           (2000, 128)          0           ['act[0][0]']                    
                                                                                                  
 dense (QDense)                 (2000, 16)           2064        ['flatten_15[0][0]']             
                                                                                                  
 input_62 (InputLayer)          [(2000, 1)]          0           []                               
                                                                                                  
 input_63 (InputLayer)          [(2000, 1)]          0           []                               
                                                                                                  
 concatenate_19 (Concatenate)   (2000, 18)           0           ['dense[0][0]',                  
                                                                  'input_62[0][0]',               
                                                                  'input_63[0][0]']               
                                                                                                  
==================================================================================================
Total params: 2,144
Trainable params: 2,144
Non-trainable params: 0
__________________________________________________________________________________________________


## Current Training Procedure
Pretraining is beneficial across machine-learning applications. Here we utilize a basic pretraining method of simply mean squared error for the first 1/3 of the total epochs. 

After training on MSE for the first 1/3 of the total epochs, we switch to the telescope loss function as implemented by C. Herwig.

