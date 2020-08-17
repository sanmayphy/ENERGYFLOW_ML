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

os.environ["CUDA_VISIBLE_DEVICES"]="0"
cuda_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )
print (torch.cuda.get_device_name( torch.cuda.current_device() ))


nameSourceFile = str(sys.argv[1])
#nameFile = str(sys.argv[2])

# %%
file_path = '/storage/agrp/sanmay/PFLOW_ANALYSIS/RectangularGeo/CROSS_CHECK/TOPO_FILES/Outputfile_V1_Samples'+nameSourceFile+'.hdf5'


f = h5py.File(file_path, 'r')

test_data_input = f['test']['input'][:]

test_data_target = f['test']['output'][:]

test_dataset = TensorDataset( Tensor(  torch.from_numpy(test_data_input).float() ), Tensor( torch.from_numpy(test_data_target).float() ) )


print('Total train : ', len(test_dataset) )
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=30)


model = SuperResModel()

model = model.to(cuda_device)

model.load_state_dict(torch.load('UNET_TRAINED/model_'+nameSourceFile+'.pt'))
model.eval()


Total_E, Target_Fr, Output_Fr = [], [], []

with tqdm(test_loader, ascii=True) as tq:

    for data, target in tq:

        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
  
        output = model(data)

        Total_E.append( data.cpu().detach().numpy() )
        Target_Fr.append( target.cpu().detach().numpy() )
        Output_Fr.append( output.cpu().detach().numpy() )


Total_E, Target_Fr, Output_Fr = np.concatenate(Total_E), np.concatenate(Target_Fr), np.concatenate(Output_Fr)        


hf = h5py.File('TOPO_FILES/PredictionFile_UNet' + nameSourceFile + '_V1.h5', 'w')

hf.create_dataset('Total_E', data=Total_E)
hf.create_dataset('Target_Fr', data=Target_Fr)
hf.create_dataset('Output_Fr', data=Output_Fr)

hf.close()
