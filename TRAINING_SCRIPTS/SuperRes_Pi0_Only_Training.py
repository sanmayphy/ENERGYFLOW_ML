import uproot
import numpy as np
import pandas as pd
import tqdm

import itertools

import os, sys

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import itertools
import math
from torch.nn import init

import dgl

from dgl import DGLGraph

from dgl.nn.pytorch import KNNGraph

import dgl.function as fn
from dgl.base import DGLError

from dgl import backend as F

import h5py

from dgl.nn.pytorch import GraphConv


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nameSourceFile = str(sys.argv[1])

train_indices = {
                  '2to5' : 0,
                  '5to10': 74200, 
                  '10to15' : 74200 + 75600,
                  '15to20' : 74200 + 75600 + 74900
                }

valid_indices = {
                  '2to5' : 0,
                  '5to10': 10600, 
                  '10to15' : 10600 + 10800,
                  '15to20' : 10600 + 10800 + 10700
                }  


train_start =   train_indices[nameSourceFile]     
valid_start =   valid_indices[nameSourceFile]

# ------------ build the geometry for dataset ------------------ #
X0, L_int = 3.9, 17.4

Z_Val = [ (3*X0/2), (3*X0 + 16*X0/2), (3*X0 + 16*X0 + 6*X0/2), (3*X0 + 16*X0 + 6*X0 + 1 + 1.5*L_int/2),
          (3*X0 + 16*X0 + 6*X0 + 1 + 1.5*L_int + 4.1*L_int/2), 
          (3*X0 + 16*X0 + 6*X0 + 1 + 1.5*L_int + 4.1*L_int + 1.8*L_int/2)
        ]

graph_size = 20 # --- K value of KNN graph

layer_size = [
64,
32,
32,
16,
16,
 8,]

scale_factors = {layer_i: int(64/lsize)*int(64/lsize) for layer_i,lsize in enumerate(layer_size)}

def goes_to(l,x,y,factor):

    xs = [factor*x+i for i in range(factor)]
    ys = [factor*y+i for i in range(factor)]
    

    goes_to = np.array(list(itertools.product(xs,ys)))
    ls = l*np.ones(len(goes_to))
    return np.column_stack([ls,goes_to]).astype(int)



goes_to_dict = {}


for layer_i in range(6):

    layer_scale = int( 64/layer_size[layer_i] )
    #print(layer_scale)
    N = layer_size[layer_i]

    for cell_x in range(N):
        for cell_y in range(N):
            goes_to_dict[(layer_i,cell_x,cell_y)] = goes_to(layer_i,cell_x,cell_y,layer_scale)


#  --------------------- function to map cell index to abs value --------------- #
def Get_XYZ_val(z_idx, idx, idy, highres=False) : 
    
    if(highres) : 
        gran = layer_size[0]
    else : 
        gran = layer_size[z_idx]
    
    x_val = idx / float(gran) * 125 - 125./2
    y_val = idy / float(gran) * 125 - 125./2
    z_val = Z_Val[z_idx] 
    
    return [x_val, y_val, z_val]

# ---------------------- define the custom dataset --------------------------------- #

class SuperResDataset(Dataset):
    
    def __init__(self, filename, ndata=-1, start=0):

        self.ndata = ndata

        self.nstart = start
        
        self.file = h5py.File(filename,'r')
                
        self.evnt_sizes = self.file['Event_size'][:]
            
        self.evnt_size_highres = self.file['Event_size_HighRes'][:]
        
        self.cumsum = np.cumsum( self.evnt_sizes )
        
        self.cumsum_highres = np.cumsum( self.evnt_size_highres )
        
        #self.broadcast_factors = [scale_factors[l_i] for l_i in self.cell_layers]
        
    def __len__(self):

        if(self.ndata == -1) :
            return len(self.evnt_sizes)   
        else : 
            return self.ndata
        
        
    def __getitem__(self, idx):
        
        idx = idx + self.nstart

        if idx == 0:
            start = 0
            start_hr = 0
        else:
            start = self.cumsum[idx-1]
            start_hr = self.cumsum_highres[idx-1]
            
        end = self.cumsum[idx]
        end_hr = self.cumsum_highres[idx]
                
#         print('start : ', start)
#         print('end : ', end)
        
#         print('start_hr : ', start_hr)
#         print('end_hr : ', end_hr)
        
        cell_xyz = self.file['CellXYLayer'][start:end]
        energies = self.file['TotalEnergy'][start:end]
        neu_energies = self.file['NeutralEnergy'][start:end]
    

        cell_xyz_highres = np.concatenate([goes_to_dict[(l,x,y)] for l,x,y in cell_xyz if l < 3])
        energies_highres = self.file['TotalEnergy_HighRes'][start_hr:end_hr]
        neu_energies_highres = self.file['NeutralEnergy_HighRes'][start_hr:end_hr]


        cell_xyz_val = np.array([ Get_XYZ_val(idx[0], idx[1], idx[2]) for idx in cell_xyz ])
        cell_xyz_val = np.reshape( cell_xyz_val, (1, cell_xyz_val.shape[0], cell_xyz_val.shape[1]) )
        cell_xyz_val = torch.FloatTensor(cell_xyz_val)
        
        cell_xyz_val_hr = np.array([ Get_XYZ_val(idx[0], idx[1], idx[2], highres=True) for idx in cell_xyz_highres ])
        cell_xyz_val_hr = np.reshape( cell_xyz_val_hr, (1, cell_xyz_val_hr.shape[0], cell_xyz_val_hr.shape[1]) )
        cell_xyz_val_hr = torch.FloatTensor(cell_xyz_val_hr)
        
        
        cell_layers = cell_xyz[:,0]
        #b_factors = [scale_factors[l_i] for l_i in cell_layers]
        
        b_factors = []
        for l_i in cell_layers : 
            if(l_i < 3) : 
                b_factors.append( scale_factors[l_i] )   
            else : 
                b_factors.append( 0 )
        
        
        graph = KNNGraph(graph_size)
        
        broad_neu_energy = torch.repeat_interleave(torch.tensor(neu_energies),torch.tensor(b_factors),dim=0)
        
        g_hr = graph(cell_xyz_val_hr)
        g_hr = dgl.transform.remove_self_loop(g_hr)
        
        g_hr.ndata['broad_neu_energy'] = torch.FloatTensor(broad_neu_energy)[:, None]       
        frac_hr = torch.FloatTensor(neu_energies_highres)/torch.FloatTensor(broad_neu_energy)
        frac_hr[ torch.isnan(frac_hr) ] = 0.  
        frac_hr[ torch.where(frac_hr < 0.) ] = 0.
        g_hr.ndata['neu_frac'] = frac_hr[:, None]
        g_hr.ndata['neu_energy'] = neu_energies_highres[:, None]
        
        

        
        sample = {
            
            'gr_hr' : g_hr
            
        }
        
        return sample


# --------------------- create the batch function ---------------- #
def create_batch(batch):
    
    
    
    graph_hr = [ sample['gr_hr'] for sample in batch ]
    
    
    return dgl.batch(graph_hr)


# -------------------------------- define the GraphNet model -------------------------------- #
class GraphNet(nn.Module):
    def __init__(self, feature_dims, input_dims=1):
        super(GraphNet, self).__init__()

        scale = 2

        self.num_layers = len(feature_dims)
        
        self.conv = nn.ModuleList()
        
        for i in range(self.num_layers):
            self.conv.append( GraphConv(in_feats = feature_dims[i - 1] if i > 0 else input_dims, 
                                        out_feats = feature_dims[i])  
                            )
        
        feature_dims.reverse()
        
        for i in range(self.num_layers):
            self.conv.append( GraphConv(in_feats = feature_dims[i] , 
                                        out_feats = feature_dims[i + 1] if i < len(feature_dims)-1 else input_dims )  
                            )
            
        
    def forward(self, g) : 
        
        with g.local_scope():
        
            for i in range(self.num_layers * 2):

                h =  g.ndata['broad_neu_energy']

                h = self.conv[i]( g, h )

                g.ndata['broad_neu_energy'] = h


            out = g.ndata['broad_neu_energy']
        
        return nn.LeakyReLU( -1. )(out)
        

# ------------------- the loss function ---------- #
def LossFunction(pred, tar, neu_energy) : 
    
    neu_energy = torch.reshape(neu_energy, (neu_energy.shape[0], ) )
    pred       = torch.reshape(pred, (pred.shape[0], ) )
    tar        = torch.reshape(tar, (tar.shape[0], ) )
    
    pos = torch.where( neu_energy !=0   )
    
    neu_energy = neu_energy[pos]
    pred = pred[pos]
    tar  = tar[pos]
    
    #wt_avg = torch.sum( neu_energy.to(dev) * torch.abs( pred.to(dev) - tar.to(dev) ) ** 2  ) 
    wt_avg = torch.sum(  torch.abs( pred.to(dev) - tar.to(dev) ) ** 2  ) 
    
    #wt_avg = wt_avg / torch.sum( neu_energy.to(dev)  )
    
    return wt_avg
    


# ----------------------- the data loader ----------------- #

train_data = SuperResDataset('training_set.h5', ndata=50000, start=train_start)
valid_data = SuperResDataset('validation_set.h5', ndata=10000, start=valid_start)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True,collate_fn=create_batch)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=10, shuffle=True,collate_fn=create_batch)


model = GraphNet( [3, 5, 7, 9, 11, 13] )
model.to(dev)

opt = optim.Adam(model.parameters(), lr=1e-4)

# ---------------- Make the training loop ----------------- #

train_loss_v, valid_loss_v = [], []


# number of epochs to train the model
n_epochs = 80

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
    #scheduler.step()
    model.train() ## --- set the model to train mode -- ##

    with tqdm.tqdm(train_loader, ascii=True) as tq:
        for gr in tq:
            
            gr  = gr.to(dev) 

            opt.zero_grad()

            tar  = gr.ndata['neu_frac']
            neu_energy = gr.ndata['broad_neu_energy']
            
            pred = model( gr )
             

            loss = LossFunction( pred, tar, neu_energy )

            loss.backward()
            # perform a single optimization step (parameter update)
            opt.step()

            # update training loss
            train_loss += loss.item() * gr.batch_size

            del gr
            torch.cuda.empty_cache()


    ######################    
    # validate the model #
    ######################
    model.eval() ## --- set the model to validation mode -- ##

    with tqdm.tqdm(valid_loader, ascii=True) as tq:

        for gr in tq:
            
            gr = gr.to(dev)

            tar  = gr.ndata['neu_frac']
            neu_energy = gr.ndata['broad_neu_energy']
            
            pred = model( gr )
             

            loss = LossFunction( pred, tar, neu_energy )
            # update training loss
            valid_loss += loss.item() * gr.batch_size

            del gr
            torch.cuda.empty_cache()



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
        torch.save(model.state_dict(), 'model_GraphNet_Neutral_' + nameSourceFile + '.pt')
        valid_loss_min = valid_loss

