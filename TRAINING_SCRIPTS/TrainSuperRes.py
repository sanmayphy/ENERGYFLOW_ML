import os
import sys
import random
# import threading
import numpy as np
import pandas as pd

import h5py

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain

import math

import uproot 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from torch.utils.data import DataLoader, TensorDataset
from torch import tensor
from HDF5Dataset import HDF5Dataset
from Models import ExpandLayer,  SuperResModel
# from torch.utils.data import DataLoader, TensorDataset
from torch.utils import data
from torch import Tensor
from pathlib import Path
import glob
import time

os.environ["CUDA_VISIBLE_DEVICES"]="1"
cuda_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )
print (torch.cuda.get_device_name( torch.cuda.current_device() ))


nameSourceFile = str(sys.argv[1])
#nameFile = str(sys.argv[2])

# %%
file_path = '/storage/agrp/sanmay/PFLOW_ANALYSIS/RectangularGeo/CROSS_CHECK/Outputfile_'+nameSourceFile+'.hdf5'
# # file_path = '/Users/antonc/Documents/GitHub/Master/Data/root_files/'
# train_dataset = HDF5Dataset(file_path, TypeData='train', recursive=False)
# valid_dataset = HDF5Dataset(file_path, TypeData='valid', recursive=False) 
# print (train_dataset.file_path)


f = h5py.File(file_path, 'r')

train_data_input = f['train']['input'][:]
train_data_target = f['train']['output'][:]

valid_data_input = f['valid']['input'][:]
valid_data_target = f['valid']['output'][:]

train_dataset = TensorDataset( Tensor(  torch.from_numpy(train_data_input).float() ), Tensor( torch.from_numpy(train_data_target).float() ) )
valid_dataset = TensorDataset( Tensor(  torch.from_numpy(valid_data_input).float() ),   Tensor( torch.from_numpy(valid_data_target).float()   ) )


print('Total train : ', len(train_dataset) )

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=30)

valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=30)



def LossFunction(pred, tar, total) : 
    

        
    pos = torch.where( total != 0 )
    
    total = total[:, 0:6, :, :]
    
    loc = torch.where( total >= 0. )
    total = total[loc]
    pred = pred[loc]
    tar = tar[loc]
    
#     target_energy = tar * total

    wt_avg = torch.sum( total.cuda() * torch.abs( pred.cuda() - tar.cuda() ) **2  )
    #print('Before div : ', wt_avg)
    wt_avg = wt_avg / torch.sum( total.cuda()  )
    
    return wt_avg


model = SuperResModel()
model.to(cuda_device)

### ----- Define the optimizer here ------ ###
optimizer = optim.Adam(model.parameters(), lr=0.0001)
total_step = len(train_loader)



# In[ ]:


train_loss_v, valid_loss_v = [], []


# In[ ]:


# number of epochs to train the model
n_epochs = 200

valid_loss_min = np.Inf # track change in validation loss

if( len(valid_loss_v) > 0 ) : 
    valid_loss_min = np.min( np.array(valid_loss_v) )

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################
    model.train() ## --- set the model to train mode -- ##
    for data, target in train_loader:
        # move tensors to GPU if CUDA is available

        #print('data shape : ', data.shape)
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
 
        loss = LossFunction(output, target, data)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)

        
    ######################  
    
    
    
    ######################    
    # validate the model #
    ######################
    model.eval()
    for data, target in valid_loader:
        # move tensors to GPU if CUDA is available
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = LossFunction(output, target, data)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
    
    # calculate average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    train_loss_v.append(train_loss) 
    valid_loss_v.append(valid_loss)
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_' + nameSourceFile + '.pt')
        valid_loss_min = valid_loss
