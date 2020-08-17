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
from Models import ExpandLayer
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

true_pix, orig_pix, det_size = 128, 64, 1250.

LayerPix = np.array([64, 32, 32, 16, 16, 8])


class Upconv(nn.Module):
    def __init__(self, upscale_factor):
        super(Upconv, self).__init__()

        self.prelu = nn.LeakyReLU(negative_slope=0.1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(256)

        self.conv1 = nn.Conv2d(1, 16,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(16, 32,kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(32, 64,kernel_size=3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(64, 128,kernel_size=3,stride=1,padding=1)
        self.conv5 = nn.Conv2d(128, 128,kernel_size=3,stride=1,padding=1)
        self.conv6 = nn.Conv2d(128, 256,kernel_size=3,stride=1,padding=1)
        self.conv7 = nn.ConvTranspose2d(256, 1,kernel_size=4,stride=2,padding=1,bias=False)
        self.conv8 = nn.PixelShuffle(upscale_factor)
        self.convf = nn.Conv2d( int(256/upscale_factor**2), 1,kernel_size=1)



    def forward(self, x):
        x = self.bn1(self.prelu(self.conv1(x)))
        x = self.bn2(self.prelu(self.conv2(x)))
        x = self.bn3(self.prelu(self.conv3(x)))
        x = self.bn4(self.prelu(self.conv4(x)))
        x = self.bn5(self.prelu(self.conv5(x)))
        x = self.bn6(self.prelu(self.conv6(x)))
        #print('Shape pre upsampling : ', x.shape)
        x = self.conv8(x)
        x = self.convf(x)
        x = self.relu(x)
        #x.dtype = torch.float64
        #print('Shape post upsampling : ', x.shape)

        return x


class SuperResModel(nn.Module):

    def __init__(self):
        super(SuperResModel, self).__init__()
        self.cnn1 = Upconv(upscale_factor = int(orig_pix/LayerPix[0]) ).float()
        self.cnn2 = Upconv(upscale_factor = int(orig_pix/LayerPix[1]) ).float()
        self.cnn3 = Upconv(upscale_factor = int(orig_pix/LayerPix[2]) ).float()
        self.cnn4 = Upconv(upscale_factor = int(orig_pix/LayerPix[3]) ).float()
        self.cnn5 = Upconv(upscale_factor = int(orig_pix/LayerPix[4]) ).float()
        self.cnn6 = Upconv(upscale_factor = int(orig_pix/LayerPix[5]) ).float()

        self.cnn_comb = nn.Conv2d(7, 6,kernel_size=1,stride=1,padding=0)


        self.cnn1_out = nn.Conv2d(1, 1,kernel_size=int(orig_pix/LayerPix[0]),stride=int(orig_pix/LayerPix[0]),padding=0)
        self.cnn2_out = nn.Conv2d(1, 1,kernel_size=int(orig_pix/LayerPix[1]),stride=int(orig_pix/LayerPix[1]),padding=0)
        self.cnn3_out = nn.Conv2d(1, 1,kernel_size=int(orig_pix/LayerPix[2]),stride=int(orig_pix/LayerPix[2]),padding=0)
        self.cnn4_out = nn.Conv2d(1, 1,kernel_size=int(orig_pix/LayerPix[3]),stride=int(orig_pix/LayerPix[3]),padding=0)
        self.cnn5_out = nn.Conv2d(1, 1,kernel_size=int(orig_pix/LayerPix[4]),stride=int(orig_pix/LayerPix[4]),padding=0)
        self.cnn6_out = nn.Conv2d(1, 1,kernel_size=int(orig_pix/LayerPix[5]),stride=int(orig_pix/LayerPix[5]),padding=0)



    def forward(self, x):


        out1 = self.cnn1( x[:,0:1, 0:LayerPix[0], 0:LayerPix[0]].float() )

        out2 = self.cnn2( x[:,1:2, 0:LayerPix[1], 0:LayerPix[1]].float() )

        out3 = self.cnn3( x[:,2:3, 0:LayerPix[2], 0:LayerPix[2]].float() )

        out4 = self.cnn4( x[:,3:4, 0:LayerPix[3], 0:LayerPix[3]].float() )

        out5 = self.cnn5( x[:,4:5, 0:LayerPix[4], 0:LayerPix[4]].float() )

        out6 = self.cnn6( x[:,5:6, 0:LayerPix[5], 0:LayerPix[5]].float() )

        out7 = x[:, 6:7, 0:orig_pix, 0:orig_pix].float()

        merged_im = torch.cat(  (out1, out2, out3, out4, out5, out6, out7), dim=1 )

        out_int = self.cnn_comb(merged_im)

        out1_f = ExpandLayer(  self.cnn1_out(out_int[:, 0:1, :, :])  )
        out2_f = ExpandLayer(  self.cnn2_out(out_int[:, 1:2, :, :])  )
        out3_f = ExpandLayer(  self.cnn3_out(out_int[:, 2:3, :, :])  )
        out4_f = ExpandLayer(  self.cnn4_out(out_int[:, 3:4, :, :])  )
        out5_f = ExpandLayer(  self.cnn5_out(out_int[:, 4:5, :, :])  )
        out6_f = ExpandLayer(  self.cnn6_out(out_int[:, 5:6, :, :])  )


        merged =  torch.cat(  (out1_f, out2_f, out3_f, out4_f, out5_f, out6_f), dim=1 )


        return merged


file_path = '/storage/agrp/sanmay/PFLOW_ANALYSIS/RectangularGeo/CROSS_CHECK/TOPO_FILES/Outputfile_V1_Samples'+nameSourceFile+'.hdf5'


f = h5py.File(file_path, 'r')

test_data_input = f['test']['input'][:]

test_data_target = f['test']['output'][:]

test_dataset = TensorDataset( Tensor(  torch.from_numpy(test_data_input).float() ), Tensor( torch.from_numpy(test_data_target).float() ) )


print('Total train : ', len(test_dataset) )
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=30)


model = SuperResModel()

model = model.to(cuda_device)

model.load_state_dict(torch.load('./model_ConvNet_'+nameSourceFile+'.pt'))
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


hf = h5py.File('TOPO_FILES/PredictionFile_ConvNet_' + nameSourceFile + '_V1.h5', 'w')

hf.create_dataset('Total_E', data=Total_E)
hf.create_dataset('Target_Fr', data=Target_Fr)
hf.create_dataset('Output_Fr', data=Output_Fr)

hf.close()
