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

test_indices = {
                  '2to5' : 0,
                  '5to10': 21200, 
                  '10to15' : 21200 + 21600,
                  '15to20' : 21200 + 21600 + 21400
                }

test_start =   test_indices[nameSourceFile]

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
        
        Trk_X_indx = self.file['Trk_X_indx'][idx:idx+1]
        Trk_Y_indx = self.file['Trk_Y_indx'][idx:idx+1]

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


        g = graph(cell_xyz_val)
        g = dgl.transform.remove_self_loop(g)
        g.ndata['neu_energy'] = neu_energies[:, None]
        g.ndata['cell_xyz'] = cell_xyz
        
        broad_neu_energy = torch.repeat_interleave(torch.tensor(neu_energies),torch.tensor(b_factors),dim=0)
        
        g_hr = graph(cell_xyz_val_hr)
        g_hr = dgl.transform.remove_self_loop(g_hr)
        
        g_hr.ndata['broad_neu_energy'] = torch.FloatTensor(broad_neu_energy)[:, None]       
        frac_hr = torch.FloatTensor(neu_energies_highres)/torch.FloatTensor(broad_neu_energy)
        frac_hr[ torch.isnan(frac_hr) ] = 0.  
        frac_hr[ torch.where(frac_hr < 0.) ] = 0.
        g_hr.ndata['neu_frac'] = frac_hr[:, None]
        g_hr.ndata['neu_energy'] = neu_energies_highres[:, None]
        g_hr.ndata['cell_xyz_highres'] = cell_xyz_highres
        g_hr.ndata['broad_factor'] = torch.repeat_interleave(torch.tensor(b_factors),torch.tensor(b_factors),dim=0)[:, None] 
        
        #print('Neu energy shape : ', cell_xyz[ np.where(cell_xyz[:,0] < 6) ].shape)

        
        

        
        sample = {
            
            'gr' : g,
            'gr_hr' : g_hr,
            'Trk_X_indx' : Trk_X_indx,
            'Trk_Y_indx' : Trk_Y_indx
            
        }
        
        return sample


# --------------------- create the batch function ---------------- #
def create_batch(batch):
    
    graph = [ sample['gr'] for sample in batch ]
    
    graph_hr = [ sample['gr_hr'] for sample in batch ]
    
    Trk_X_indx = np.array([ sample['Trk_X_indx'] for sample in batch ])
    Trk_Y_indx = np.array([ sample['Trk_Y_indx'] for sample in batch ])
    
    return dgl.batch(graph), dgl.batch(graph_hr), Trk_X_indx, Trk_Y_indx


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
        


# --------- node to image function --------------- #
def MakeImage( energies, cell_xyz, out_l = 6 ) : 
    
    img = torch.zeros([out_l, 64, 64], device=dev)

    energies = torch.reshape(energies, (energies.shape[0], ) )
    
    img[ cell_xyz[:,0], cell_xyz[:,1], cell_xyz[:,2] ] = energies
    
    return img.cpu().detach().numpy()

# ----------------------- the data loader ----------------- #

test_data = SuperResDataset('test_set.h5', ndata=3000, start=test_start)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=True,collate_fn=create_batch)



# ---------------- declare the model & optimizer ----------------- #
model = GraphNet([3, 5, 7, 9, 11, 13])
model.to(dev)


model.load_state_dict(torch.load('model_GraphNet_Neutral_' + nameSourceFile + '.pt'))
model.eval()


Pred_En_HighRes, Truth_En_HighRes, Truth_En_LowRes = [], [], []
Trk_X_indx, Trk_Y_indx = [], []

with tqdm.tqdm(test_loader, ascii=True) as tq:

    for g, gr, Trk_x_indx, Trk_y_indx in tq:
        
        gr = gr.to(dev)
        g  = g.to(dev)

        #hout = model( gr )


        # ------- create the graph list ---------- #
        with gr.local_scope():
         with g.local_scope() : 
            # node_list = gr.batch_num_nodes
            # node_list.insert(0, 0)
            # cum_node = np.cumsum( node_list )

            g_list = dgl.unbatch(g)
            graph_list = dgl.unbatch(gr)
            
            batch_len = len(graph_list)

            for ig in range(batch_len) :

                g_i    = g_list[ig]
                gr_i   = graph_list[ig]  

                hout = model( gr_i )

                pred_i = hout#hout[ cum_node[ig] : cum_node[ig+1] ]    
                tar_i  = gr_i.ndata['neu_frac']

                neu_energy = gr_i.ndata['broad_neu_energy']
                broad_factor = gr_i.ndata['broad_factor']

                cell_xyz_highres = gr_i.ndata['cell_xyz_highres'] 
                # -------------------------------------------- #
                neu_energy_l = g_i.ndata['neu_energy'] 
                cell_xyz = g_i.ndata['cell_xyz'] 

                img_truth_ne_lr = MakeImage( neu_energy_l, cell_xyz, out_l = 6 ) 
                Truth_En_LowRes.append( img_truth_ne_lr )
                # ---------------------------------------------#
                tar_ne_en_hr  = tar_i  * neu_energy 
                pred_ne_en_hr = pred_i * neu_energy 

                img_truth_ne_en = MakeImage( tar_ne_en_hr, cell_xyz_highres, out_l = 3 )
                img_pred_ne_en = MakeImage( pred_ne_en_hr, cell_xyz_highres, out_l = 3 )

                Truth_En_HighRes.append( img_truth_ne_en )
                Pred_En_HighRes.append( img_pred_ne_en )

                Trk_X_indx.append(Trk_x_indx[ig])
                Trk_Y_indx.append(Trk_y_indx[ig])


Trk_X_indx = np.array(Trk_X_indx)
Trk_Y_indx = np.array(Trk_Y_indx)

Truth_En_HighRes = np.array( Truth_En_HighRes )
Pred_En_HighRes = np.array( Pred_En_HighRes )
Truth_En_LowRes = np.array(Truth_En_LowRes)


TR_EN = np.array( [ np.sum(im) for im in Truth_En_HighRes ] )
PR_EN = np.array( [ np.sum(im) for im in Pred_En_HighRes ] )

TR_EN = TR_EN[ np.where(TR_EN!=0) ]
PR_EN = PR_EN[ np.where(TR_EN!=0) ]
#print('Total PR_EN : ', np.sum(PR_EN) )

print( 'Avg Er : ', np.mean( (PR_EN )/TR_EN ) )
print( 'Sigma Er : ', np.std( (PR_EN)/TR_EN ) )


hf = h5py.File('PredictionFile_SR_PI0_' + nameSourceFile + '.h5', 'w')

hf.create_dataset('Truth_En_HighRes', data=Truth_En_HighRes, compression = "lzf")
hf.create_dataset('Pred_En_HighRes', data=Pred_En_HighRes, compression = "lzf")
hf.create_dataset('Truth_En_LowRes', data=Truth_En_LowRes, compression = "lzf")
hf.create_dataset('Trk_X_indx', data=Trk_X_indx, compression = "lzf")
hf.create_dataset('Trk_Y_indx', data=Trk_Y_indx, compression = "lzf")

hf.close() 
