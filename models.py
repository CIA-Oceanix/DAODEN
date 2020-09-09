#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Duong Nguyen
# Created Date: 2019/03/03
# =============================================================================
"""Models for DAODEN"""


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (1920,1080)
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pickle
import sys
import os
#from tqdm import tqdm
import time

#===================================================================================
class BiNN(torch.nn.Module):
    """ Implementation of a Bi-linear Neural Network (BiNN)
    https://ieeexplore.ieee.org/document/8553492
    
    The BiNN is an architecture designed specifically for Geophysical Dynamics. It
    contains bi-linear terms.
    """
    def __init__(self,
                 input_dim, 
                 output_dim, 
                 hidden_linear_dim, 
                 n_bilinear_layers, 
                 n_trans_layers = 1,
                 max_value = 10000.,
                 device = torch.device("cpu")
                ):
        super(BiNN, self).__init__()
        self.input_dim         = input_dim
        self.output_dim        = output_dim
        self.hidden_linear_dim = hidden_linear_dim
        self.n_bilinear_layers = n_bilinear_layers
        self.n_trans_layers    = n_trans_layers 
        self.max_value         = torch.tensor(max_value).to(device)

        self.linearCell   = torch.nn.Linear(self.input_dim, self.hidden_linear_dim)
        self.BlinearCell1 = torch.nn.ModuleList([torch.nn.Linear(self.input_dim, 1) for i in range(self.n_bilinear_layers)])
        self.BlinearCell2 = torch.nn.ModuleList([torch.nn.Linear(self.input_dim, 1) for i in range(self.n_bilinear_layers)])
        augmented_size    = self.n_bilinear_layers + self.hidden_linear_dim
        self.transLayers = torch.nn.ModuleList([torch.nn.Linear(augmented_size, self.output_dim)])
        self.transLayers.extend([torch.nn.Linear(self.output_dim, self.output_dim) for i in range(1, self.n_trans_layers)])
        self.outputLayer  = torch.nn.Linear(self.output_dim, self.output_dim)

    def forward(self, inp, dt):
        L_outp   = self.linearCell(inp)
        l_BP_outp = [(self.BlinearCell1[i](inp)[:,0]*self.BlinearCell2[i](inp)[:,0]).unsqueeze(1) for i in
                                               range((self.n_bilinear_layers))]

        aug_vect = torch.cat([L_outp]+l_BP_outp, dim=1)
        for i in range((self.n_trans_layers)):
            aug_vect = (self.transLayers[i](aug_vect))
        grad = self.outputLayer(aug_vect)
        
        return torch.max(torch.min(grad, self.max_value), -self.max_value)
#model  = FC_net(params)

#===================================================================================
class BINN_convnet(torch.nn.Module):
    """ Implementation of a convolutional version of Bi-linear Neural Network (BiNN)
    http://centaur.reading.ac.uk/89798/
    """
    def __init__(self, params, device = torch.device("cpu")):
        super(BINN_convnet, self).__init__()
        w = np.random.uniform(size=(self.output_dim,3))
        self.w = torch.nn.Parameter(torch.from_numpy(w).float())
        self.linearCell = torch.nn.Linear(self.input_dim+params['dim_latent'], self.output_dim) 
        self.device = device
        #torch.nn.init.uniform_(self.m)
        self.dim_output = self.output_dim
    def forward(self, inp, dt):
        aug_inp = inp
        x = torch.zeros((inp.shape[0],self.dim_output)).to(self.device);
        x[:,0] = (self.w[0,0]*aug_inp[:,1]-self.w[0,1]*aug_inp[:,self.dim_output-2])*self.w[0,2]*aug_inp[:,self.dim_output-1]
        x[:,1] = (self.w[1,0]*aug_inp[:,2]-self.w[1,1]*aug_inp[:,self.dim_output-1])*self.w[1,0]*aug_inp[:,0]
        x[:,self.dim_output-1] = (self.w[-1,0]*aug_inp[:,0]-self.w[-1,1]*aug_inp[:,self.dim_output-3])*self.w[-1,2]*aug_inp[:,self.dim_output-2]
        for j in range(2,self.dim_output-1):
            x[:,j] = (self.w[j,0]*aug_inp[:,j+1]-self.w[j,1]*aug_inp[:,j-2])*self.w[j,2]*aug_inp[:,j-1]
        grad = x + self.linearCell(aug_inp);
        return grad

#===================================================================================
class RK4_Net(torch.nn.Module):
    """ Neural network implementation of the Runge-Kutta 4 (RK4) integration scheme
    """
    def __init__(self, dyn_model, dt_integration, max_value = 1000.,device = torch.device("cpu")):
        super(RK4_Net, self).__init__()
#            self.add_module('Dyn_net',FC_net(params))
        self.device = device
        self.Dyn_net = dyn_model
        self.dt_integration = dt_integration
        self.max_value         = torch.tensor(max_value).to(device)
    def forward(self, inp, dt):
        #grad = torch.max(torch.min(self.Dyn_net(inp,dt), self.max_value), -self.max_value)
        k1   = torch.max(torch.min(self.Dyn_net(inp,dt), self.max_value), -self.max_value)
        k2   = torch.max(torch.min(self.Dyn_net(inp+0.5*self.dt_integration*k1,dt), self.max_value), -self.max_value)
        k3   = torch.max(torch.min(self.Dyn_net(inp+0.5*self.dt_integration*k2,dt), self.max_value), -self.max_value)
        k4   = torch.max(torch.min(self.Dyn_net(inp+self.dt_integration*k3,dt), self.max_value), -self.max_value)
        pred = torch.max(torch.min(inp +dt*(k1+2*k2+2*k3+k4)/6, self.max_value), -self.max_value)
        return pred, k1
#RINN_model = INT_net(params)
#RINN_model = RINN_model.to(device)



#===================================================================================
def MLP(l_layer_sizes,
        activation=nn.ReLU(),
        activate_final=False,
        use_bias=True,
        use_dropout=False):
    l_modules = []
    for d_i in range(len(l_layer_sizes)-2):
        l_modules.append(nn.Linear(l_layer_sizes[d_i],l_layer_sizes[d_i+1],bias = use_bias))
        l_modules.append(activation)
    # Output layer
    l_modules.append(nn.Linear(l_layer_sizes[-2],l_layer_sizes[-1], bias = use_bias))
    if activate_final:
        l_modules.append(activation)
    return nn.Sequential(*l_modules)

#===================================================================================
def encode_all(inputs, encoder):
    """Encodes a timeseries of inputs with a time independent encoder.
    Args:
        inputs: A [time, batch, feature_dimensions] tensor.
        encoder: A network that takes a [batch, features_dimensions] input and
              encodes the input.
    Returns:
        A [time, batch, encoded_feature_dimensions] output tensor.
    """
    input_shape = inputs.shape
    num_timesteps, batch_size, features_dimensions = input_shape[0], input_shape[1], input_shape[2]
    reshaped_inputs = inputs.contiguous().view(-1, features_dimensions)
    inputs_encoded = encoder(reshaped_inputs).view(num_timesteps, batch_size, -1)
    return inputs_encoded

#===================================================================================
def reverse_sequence(seq,seq_axis=0,device=torch.device("cpu")):
    """Reserves a timeseries.
    Args:
        seq: A [time, batch, feature_dimensions] tensor.
    Returns:
        A [time, batch, encoded_feature_dimensions] output tensor.
    """
    v_inv_idx = torch.arange(seq.size(seq_axis)-1, -1, -1).long().to(device)
    return seq.index_select(seq_axis,v_inv_idx)
