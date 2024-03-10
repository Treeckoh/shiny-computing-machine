# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 16:13:06 2024

@author: Tom
"""
import tensorflow as tf
class ConvPoolBatchN(tf.keras.layers.Layer):
    '''
    Layer class to do a 1d/2d conv+maxpool+batchnorm block
    Args:
        dimensions (int): 1-3 for what dimension of convolution and maxpool to do 
                          Eg 1 -> Conv1D MaxPool1D
        conv_kernel (int/tuple): dimensions for the kernel size in the convolution layer 
                                 Eg 3 ->(3,3) in 1D
        pool_kernel (int/tuple): dimensions for the pool size in the MaxPool layer 
                                 Eg 2 -> (2,2) in 1D
        activation (str): The activation function to use for the convolution layer 
                          Default to 'relu' for max(0,x)
    
    Examplewith default params and dimensions 1, filters 10:
    
    ConvPoolBatchN(dimensions = 1, n_filters = 10) 
    
    Will effectively add the layers below to a network:
    
    inputs = inputs
    x = Conv1D(10,3, activation = 'relu')(x)
    x = MaxPool1D(2)(x)
    x = BatchNormalization()(x)
    return x
    
    Or with sequential
    
    tf.keras.Sequential([
        Conv1D(10,3, activation = 'relu'),
        MaxPool1D(2),
        BatchNormalization()
    ])
    
    '''
    def __init__(self,
                 dimensions,
                 n_filters,
                 conv_kernel = 3,
                 pool_kernel = 2,
                 activation = 'relu',
                 **kwargs):
        self.dimensions = dimensions
        self.n_filters = n_filters
        self.conv_kernel = conv_kernel
        self.pool_kernel = pool_kernel
        self.activation = activation
        self.create_layers()
        super().__init__(**kwargs)
    def create_layers(self):
        '''
        Creates a self.x_layer for the layers in the block
        Will check the sself.dimensions variable to check which size conv and pool to use
        '''
        self.layers = []
        if self.dimensions == 1:
            self.layers.append(tf.keras.layers.Conv1D(filters = self.n_filters, kernel_size = self.conv_kernel, activation = self.activation))
            self.layers.append(tf.keras.layers.MaxPool1D(pool_size = self.pool_kernel))
            self.layers.append(tf.keras.layers.BatchNormalization())
            
        elif self.dimensions == 2:
            self.layers.append(tf.keras.layers.Conv2D(filters = self.n_filters, kernel_size = self.conv_kernel, activation = self.activation))
            self.layers.append(tf.keras.layers.MaxPool2D(pool_size = self.pool_kernel))
            self.layers.append(tf.keras.layers.BatchNormalization())
            
        elif self.dimensions == 3:
            self.layers.append(tf.keras.layers.Conv3D(filters = self.n_filters, kernel_size = self.conv_kernel, activation = self.activation))
            self.layers.append(tf.keras.layers.MaxPool3D(pool_size = self.pool_kernel))
            self.layers.append(tf.keras.layers.BatchNormalization())
            
    def call(self, inputs):
        '''
        Uses the call function of the layer to return ConvND, MaxPoolND, BatchNormalization layers on the inputs
        Args:
            inputs (tensorflow tensor): Likely the training data being run through the Neural Network
        
        Uses the tensorflow functional API to feed the results through the block to get the output
        '''
        x = self.layers[0](inputs)
        for layer in self.layers[1:]:
            x = layer(x)
        return x
        
if __name__ == '__main__':
    
    test_cpb_layer = ConvPoolBatchN(dimensions = 2, 
                                    n_filters = 32)
    
    print(test_cpb_layer.layers)