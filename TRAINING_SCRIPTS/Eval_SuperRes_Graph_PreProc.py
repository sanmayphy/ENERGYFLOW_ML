import uproot
import numpy as np
import pandas as pd
import tqdm

import itertools

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

# ------------ build the geometry for dataset ------------------ #
X0, L_int = 3.9, 17.4

Z_Val = [ (3*X0/2), (3*X0 + 16*X0/2), (3*X0 + 16*X0 + 6*X0/2), (3*X0 + 16*X0 + 6*X0 + 1 + 1.5*L_int/2),
          (3*X0 + 16*X0 + 6*X0 + 1 + 1.5*L_int + 4.1*L_int/2), 
          (3*X0 + 16*X0 + 6*X0 + 1 + 1.5*L_int + 4.1*L_int + 1.8*L_int/2)
        ]

graph_size = 10 # --- K value of KNN graph

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
    
    def __init__(self, filename, ndata=-1):

        self.ndata = ndata
        
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
              
        Total = np.sum(energies)


        energies = energies/Total
        neu_energies = neu_energies/Total
        
        cell_xyz_highres = np.concatenate([goes_to_dict[(l,x,y)] for l,x,y in cell_xyz if l < 3])
        energies_highres = self.file['TotalEnergy_HighRes'][start_hr:end_hr]
        neu_energies_highres = self.file['NeutralEnergy_HighRes'][start_hr:end_hr]

        energies_highres = energies_highres/Total
        neu_energies_highres = neu_energies_highres/Total
        
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
        g.ndata['energy'] = torch.reshape(torch.FloatTensor(energies), (  torch.FloatTensor(energies).shape[0],1 ) )
        g.ndata['broadcast'] = torch.tensor(b_factors)
        
        g.ndata['parent_node'] = g.number_of_nodes() * torch.ones([ g.number_of_nodes() ], dtype=torch.int)
        g.ndata['_ID'] = g.nodes()
        g.ndata['cell_xyz'] = cell_xyz
        g.ndata['neu_energy'] = neu_energies[:, None]
        
        g_hr = graph(cell_xyz_val_hr)
        g_hr = dgl.transform.remove_self_loop(g_hr)
        
        g_hr.ndata['energy'] = torch.FloatTensor(energies_highres)       
        frac_hr = torch.FloatTensor(neu_energies_highres)/torch.FloatTensor(energies_highres)
        frac_hr[ torch.isnan(frac_hr) ] = 0.  
        frac_hr[ torch.where(frac_hr < 0.) ] = 0.
        g_hr.ndata['neu_frac'] = frac_hr[:, None]
        g_hr.ndata['neu_energy'] = neu_energies_highres[:, None]
        #g_hr.ndata['cell_xyz_highres'] = cell_xyz_highres
        
        g_out_hr = graph(cell_xyz_val_hr)
        g_out_hr = dgl.transform.remove_self_loop(g_out_hr)
        g_out_hr.ndata['cell_xyz_highres'] = cell_xyz_highres
        
        pi0_phi = self.file['Pi0_Phi'][idx:idx+1]
        pi0_theta = self.file['Pi0_Theta'][idx:idx+1]
        
        sample = {
            'gr' : g,
            'gr_hr' : g_hr,
            'gr_out_hr' : g_out_hr,
            'pi0_theta' : torch.FloatTensor( np.cos(pi0_theta) ),
            'pi0_phi' : torch.FloatTensor( np.cos(pi0_phi) ),
            'Total_E' : Total
#             'cell_xyz' : cell_xyz,
#             'cell_xyz_highres' : cell_xyz_highres
            
            
        }
        
        return sample


# --------------------- create the batch function ---------------- #
def create_batch(batch):
    
    
    graph = [ sample['gr'] for sample in batch ]
    graph_hr = [ sample['gr_hr'] for sample in batch ]
    graph_out_hr = [ sample['gr_out_hr'] for sample in batch ]
    
    pi0_theta = torch.tensor([ sample['pi0_theta'] for sample in batch ])
    pi0_phi = torch.tensor([ sample['pi0_phi'] for sample in batch ])

    Total_E = torch.tensor([ sample['Total_E'] for sample in batch ])
    
    
    return dgl.batch(graph), dgl.batch(graph_hr), dgl.batch(graph_out_hr), pi0_theta, pi0_phi, Total_E


# ================== STARTING DIFFERENT PARTS OF UNET MODEL ============================== #

# ------------- make the down conv layer ---------- #
class TopKPooling(nn.Module):
    def __init__(self, frac, in_feat=1, out_feat=1):
        super(TopKPooling, self).__init__()

        self.p = nn.Parameter(torch.Tensor(in_feat, out_feat))
        self.reset_parameters()
        
        self.frac = frac
        
        self.in_feat = in_feat
        self.out_feat = out_feat
        
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.p is not None:
            init.xavier_uniform_(self.p)


    def forward(self, gr):
        
        output_gr = []
        
        with gr.local_scope():
            
            graph_list = dgl.unbatch(gr)
            
            batch_itr = 0
              
            for g in graph_list : 
                    
                    batch_itr += 1
                    k_val = int(g.number_of_nodes() * self.frac)
                    
                    X = g.ndata['energy']
                    
                    #print('X shape : ', X.shape)
                    
                    # ----- y = X . p / ||p||
                    if(self.p.shape[1] == 1) : 
                        y = (X * self.p)/torch.sqrt( torch.sum(self.p ** 2) ).item()
                    else : 
                        y = torch.mm(X, self.p)/torch.sqrt( torch.sum(self.p ** 2) ).item()
                    
                    #print('y shape : ', y.shape)
                    g.ndata['y'] = y #torch.transpose(y, 1, 0) 
                    
                    
                    # ------ idx = rank(y, k) ----------- #
                    pooled_node_features, selected_nodes = dgl.topk_nodes(g, 'y', k=k_val, descending=True, idx=0)
                    
                    # --- reduced representation ----- #
                    sg = g.subgraph( selected_nodes[0].tolist() )
                    sg.copy_from_parent()
                    
                    X_bar = sg.ndata['energy']
                    
                    y_bar = nn.Sigmoid()(pooled_node_features)
                    
                    
                    X_bar = torch.reshape( X_bar, ( X_bar.shape[0], self.out_feat) )
#                     print('X_bar shape : ', X_bar.shape)
#                     print('y_bar shape : ', y_bar.shape)
                    
                    
                    mod_en = X_bar * y_bar[0]
                    
                    #print('Mod en shape : ', mod_en.shape)
                    
                    sg.ndata['energy'] = mod_en

                    sg.ndata['parent_node'] = sg.parent.number_of_nodes() * torch.ones([ sg.number_of_nodes() ], dtype=torch.int)
                    sg.ndata['own_node'] = sg.number_of_nodes() * torch.ones([ sg.number_of_nodes() ], dtype=torch.int64)
                    sg.ndata['selected_node'] = selected_nodes[0]
                    
                    
                    output_gr.append( sg )
                    
                    #print('End batch itr : ', batch_itr)
        
        
        #print(output_gr)
        return dgl.batch(output_gr)

# ------------------------------ define the UpPool block -------------------------- #
class UpPool(nn.Module):
    def __init__(self, feat_dim):
        super(UpPool, self).__init__()

        self.dim = feat_dim
        
    def forward(self, bg, bg_u):

        output_gr = []
        
        with bg.local_scope():
            with bg_u.local_scope():
                
                graph_list = dgl.unbatch(bg)
                graph_list_u = dgl.unbatch(bg_u)

                for ig in range(len(graph_list)) :

                    g = graph_list[ig]
                    g_u = graph_list_u[ig]
                    
                    #print('----- start filling -------')
                    n_unpooled_node = g.ndata['parent_node'][0]

                    selected_nodes = g.ndata['_ID'][:, None] #a
                    pooled_node_features = g.ndata['energy'] #b
                    
#                     print('selected_nodes shape : ', selected_nodes.shape)
#                     print('pooled_node_features : ', pooled_node_features.shape)

                    expanded_node = selected_nodes.expand_as(pooled_node_features) #c
                    expanded_node = expanded_node.to(dev)

                    pooled_node_features = pooled_node_features.to(dev)

                    x = torch.zeros(n_unpooled_node, self.dim, device=dev)
                    #x.to(dev)
                    
                    x.scatter_(0,  expanded_node, pooled_node_features )
                    
                    #print('----- end filling -------')
                    
                    g_new = dgl.DGLGraph()
                    g_new.add_nodes(n_unpooled_node)
                    
                    src, dst = g_u.edges()
                    g_new.add_edges(src, dst)
                    
                    g_new.ndata['energy'] = x
                    g_new.ndata['parent_node'] = g_u.ndata['parent_node'][0] * torch.ones([ g_new.number_of_nodes() ], dtype=torch.int)
                    g_new.ndata['_ID'] = torch.tensor(g_u.ndata['_ID'], dtype=torch.int64)
                    
#                     print('Output node energy shape : ', g_new.ndata['energy'].shape)
                    
                    output_gr.append(g_new) 
                 
        
        return dgl.batch(output_gr)


# ------------------------- define the broadcasting ---------------- #

class Broadcasting(nn.Module):
    def __init__(self):
        super(Broadcasting, self).__init__()
        
    def forward(self, bg, bg_out_hr,feature_name='energy',out_name='neu_energy'):
        
        output_gr = []   
        
        with bg.local_scope():
            with bg_out_hr.local_scope():
                
                graph_list = dgl.unbatch(bg)
                graph_list_out_hr = dgl.unbatch(bg_out_hr)
                
                for ig in range(len(graph_list)) :
                    
                    g = graph_list[ig]
                    g_out_hr = graph_list_out_hr[ig]
                
                    data = g.ndata[feature_name] 
                    data = torch.reshape(data, (data.shape[0],) )
                    b_factors = g.ndata['broadcast']

                    out = torch.repeat_interleave(data,b_factors,dim=0)

                    g_out_hr.ndata[out_name] = out[:, None]
                    
                    output_gr.append(g_out_hr  )
        
        return dgl.batch(output_gr)


# -------------------------------- define the GraphUNet model -------------------------------- #
class GraphUNet(nn.Module):
    def __init__(self):
        super(GraphUNet, self).__init__()

        scale = 2

        self.do_conv1 = GraphConv(in_feats = 1, out_feats = 1)
        self.do_pool1 = TopKPooling(frac=0.75, in_feat=1, out_feat=1) 
        
        self.do_conv2 = GraphConv(in_feats = 1, out_feats = 5*scale)
        self.do_pool2 = TopKPooling(frac=0.75, in_feat=5*scale, out_feat=5*scale)
        
        self.do_conv3 = GraphConv(in_feats = 5*scale, out_feats = 7*scale)
        self.do_pool3 = TopKPooling(frac=0.75, in_feat=7*scale, out_feat=7*scale)
        
        self.do_conv4 = GraphConv(in_feats = 7*scale, out_feats = 9*scale)
        self.do_pool4 = TopKPooling(frac=0.75, in_feat=9*scale, out_feat=9*scale)
        
        self.bn_conv = GraphConv(in_feats = 9*scale, out_feats = 9*scale)
        
        self.up_pool1 = UpPool(feat_dim = 9*scale)
        self.up_conv1 = GraphConv(in_feats = 9*scale, out_feats = 7*scale)
        
        self.up_pool2 = UpPool(feat_dim = 7*scale)
        self.up_conv2 = GraphConv(in_feats = 7*scale, out_feats = 5*scale)
        
        self.up_pool3 = UpPool(feat_dim = 5*scale)
        self.up_conv3 = GraphConv(in_feats = 5*scale, out_feats = 1)
        
        self.up_pool4 = UpPool(feat_dim = 1)
        self.up_conv4 = GraphConv(in_feats = 1, out_feats = 1)
        
        self.broadcast = Broadcasting()
        
        self.b_conv = GraphConv(in_feats = 1, out_feats = 1)
        
        self.proj = nn.Sequential(
                    nn.Linear(1, 20),
                    nn.LeakyReLU( -0.8 ), 
                    nn.Tanh(),

                    nn.Linear(20, 40),                   
                    nn.LeakyReLU( -0.8 ), 
                    nn.Tanh(),

                    nn.Linear(40, 30),                   
                    nn.LeakyReLU( -0.8 ),
                    nn.Tanh(),

                    nn.Linear(30, 10),                   
                    nn.LeakyReLU( -0.8 ),
                    nn.Tanh(),

                    nn.Linear(10, 5),                    
                    nn.LeakyReLU( -0.8 ),
                    nn.Tanh(),

                    nn.Linear(5, 3)
                    )
        
    def forward(self, bg, bg_out_hr):

        #print(bg.batch_num_nodes)
        #print('bg en shape : ', bg.ndata['energy'].shape )
        
        # ---- 1st down block ---- #
        h = bg.ndata['energy']
        h = self.do_conv1(bg, h)
        bg.ndata['energy'] = h
        bg_d1 = self.do_pool1(bg)
        #print('bg D1 en shape : ', bg_d1.ndata['energy'].shape )
        
        # ---- 2nd down block ---- #
        h = bg_d1.ndata['energy']
        h = self.do_conv2(bg_d1, h)
        bg_d1.ndata['energy'] = h
        bg_d2 = self.do_pool2(bg_d1)
        #print('bg D2 en shape : ', bg_d2.ndata['energy'].shape )
        
        # ---- 3rd down block ---- #
        h = bg_d2.ndata['energy']
        h = self.do_conv3(bg_d2, h)
        bg_d2.ndata['energy'] = h
        bg_d3 = self.do_pool3(bg_d2)
        #print('bg D3 en shape : ', bg_d3.ndata['energy'].shape )
        
        # ---- 4th down block ---- #
        h = bg_d3.ndata['energy']
        h = self.do_conv4(bg_d3, h)
        bg_d3.ndata['energy'] = h
        bg_d4 = self.do_pool4(bg_d3)
        #print('bg D4 en shape : ', bg_d4.ndata['energy'].shape )
        
        # ------- bottle-neck block ----- #
        h = bg_d4.ndata['energy']
        h = self.bn_conv( bg_d4, h )
        bg_bn = bg_d4
        bg_bn.ndata['energy'] = h
        
        
        # --------- 1st up-conv block ------- #
        bg_u1 = self.up_pool1(bg_bn, bg_d3)
        bg_u1.ndata['energy'] = bg_u1.ndata['energy'] + bg_d3.ndata['energy'] # -- the 1st skip con -- #
        h = bg_u1.ndata['energy']
        h = self.up_conv1(bg_u1, h)
        bg_u1.ndata['energy'] = h
        #print('bg U1 en shape : ', bg_u1.ndata['energy'].shape )
        
        
        # --------- 2nd up-conv block ------- #
        bg_u2 = self.up_pool2(bg_u1, bg_d2)
        bg_u2.ndata['energy'] = bg_u2.ndata['energy'] + bg_d2.ndata['energy'] # -- the 2nd skip con -- #
        h = bg_u2.ndata['energy']
        h = self.up_conv2(bg_u2, h)
        bg_u2.ndata['energy'] = h
        #print('bg U2 en shape : ', bg_u2.ndata['energy'].shape )
        
        # --------- 3rd up-conv block ------- #
        bg_u3 = self.up_pool3(bg_u2, bg_d1)
        bg_u3.ndata['energy'] = bg_u3.ndata['energy'] + bg_d1.ndata['energy'] # -- the 3rd skip con -- #
        h = bg_u3.ndata['energy']
        h = self.up_conv3(bg_u3, h)
        bg_u3.ndata['energy'] = h
        #print('bg U3 en shape : ', bg_u3.ndata['energy'].shape )
        
        # --------- 4th up-conv block ------- #
        bg_u4 = self.up_pool4(bg_u3, bg)
        bg_u4.ndata['energy'] = bg_u4.ndata['energy'] + bg.ndata['energy'] # -- the 4th skip con -- #
        h = bg_u4.ndata['energy']
        h = self.up_conv4(bg_u4, h)
        bg_u4.ndata['energy'] = h
        #print('bg U4 en shape : ', bg_u4.ndata['energy'].shape )
        
        bg_u4.ndata['broadcast'] = bg.ndata['broadcast']
        
        bg_out_hr = self.broadcast( bg_u4, bg_out_hr )
        
        h = bg_out_hr.ndata['neu_energy']
        h = self.b_conv( bg_out_hr, h )
        
        #h = nn.Tanh()(h)
        h = nn.LeakyReLU( -1. )(h)
        bg_out_hr.ndata['neu_energy'] = h
        
        hout = []
                    
        with bg_out_hr.local_scope():
            graph_list_out_hr = dgl.unbatch(bg_out_hr)  
            
            for ig in range(len(graph_list_out_hr)) :
                    
                    g_out_hr = graph_list_out_hr[ig]
                    
                    h = g_out_hr.ndata['neu_energy']
                    
                    ho = self.proj( torch.mean(h, dim=0) )

                    hout.append(ho[None,:])
                    
        hout = torch.cat( hout, dim=0 )
        
        return bg_out_hr, hout


# --------- node to image function --------------- #
def MakeImage( energies, cell_xyz, out_l = 6 ) : 
    
    img = torch.zeros([out_l, 64, 64], device=dev)

    energies = torch.reshape(energies, (energies.shape[0], ) )
    
    img[ cell_xyz[:,0], cell_xyz[:,1], cell_xyz[:,2] ] = energies
    
    return img.cpu().detach().numpy()

# ----------------------- the data loader ----------------- #

test_data = SuperResDataset('test_set.h5', ndata=1000)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=True,collate_fn=create_batch)



# ---------------- declare the model & optimizer ----------------- #
model = GraphUNet()
model.to(dev)


model.load_state_dict(torch.load('model_GraphUNet_PreProc.pt'))
model.eval()


Total_E_LowRes, Pred_En_HighRes, Truth_En_HighRes = [], [], []

Truth_Theta, Truth_Phi, Pred_Theta, Pred_Phi = [], [], [], []

Mass_Rat = []

with tqdm.tqdm(test_loader, ascii=True) as tq:

    for gr ,gr_hr, gr_out_hr, pi0_theta, pi0_phi, Total_E in tq:
        
        gr ,gr_hr, gr_out_hr = gr.to(dev) ,gr_hr.to(dev), gr_out_hr.to(dev)

        bg, hout = model( gr, gr_out_hr )

        # ------- create the graph list ---------- #
        graph_list = dgl.unbatch(gr)
        graph_list_tru_hr = dgl.unbatch(gr_hr)
        graph_list_out_hr = dgl.unbatch(bg) 

        batch_len = len(graph_list)

        for ig in range(batch_len) :

      
            gr_lr     = graph_list[ig]      
            gr_hr_tar = graph_list_tru_hr[ig]
            gr_hr_pr  = graph_list_out_hr[ig]
            

            Total = Total_E[ig]

            cell_xyz = gr_lr.ndata['cell_xyz']
            cell_xyz_highres = gr_hr_pr.ndata['cell_xyz_highres']
   

            total_en     = gr_lr.ndata['energy'] * Total
            tar_ne_en_hr = gr_hr_tar.ndata['neu_energy'] * Total
            pred_ne_en_hr = gr_hr_pr.ndata['neu_energy'] * Total

            pi0_theta_tr = pi0_theta[ig].cpu().detach().numpy()
            pi0_phi_tr   = pi0_phi[ig].cpu().detach().numpy()

            pi0_theta_pr = hout[ig][0].cpu().detach().numpy()
            pi0_phi_pr   = hout[ig][1].cpu().detach().numpy()

            mass_rat     = hout[ig][2].cpu().detach().numpy()

            img_total = MakeImage( total_en, cell_xyz, out_l = 6 )
            img_truth_ne_en = MakeImage( tar_ne_en_hr, cell_xyz_highres, out_l = 3 )
            img_pred_ne_en = MakeImage( pred_ne_en_hr, cell_xyz_highres, out_l = 3 )

            #print( 'Pred image shape : ', img_pred_ne_en.shape )

            Total_E_LowRes.append( img_total )
            Truth_En_HighRes.append( img_truth_ne_en )
            Pred_En_HighRes.append( img_pred_ne_en )

            Truth_Theta.append( pi0_theta_tr )
            Truth_Phi.append( pi0_phi_tr ) 
            Pred_Theta.append( pi0_theta_pr ) 
            Pred_Phi.append( pi0_phi_pr )

            Mass_Rat.append( mass_rat )



Total_E_LowRes = np.array( Total_E_LowRes )
Truth_En_HighRes = np.array( Truth_En_HighRes )
Pred_En_HighRes = np.array( Pred_En_HighRes )

Truth_Theta = np.array( Truth_Theta )
Truth_Phi = np.array( Truth_Phi ) 
Pred_Theta = np.array( Pred_Theta )
Pred_Phi = np.array( Pred_Phi )

Mass_Rat = np.array( Mass_Rat )

print('Total E shape : ', Total_E_LowRes.shape )

print( 'Avg Theta: ', np.mean( (Pred_Theta - Truth_Theta)/Truth_Theta ) )
print( 'Avg Phi: ', np.mean( (Pred_Phi - Truth_Phi)/Truth_Phi ) )
print( 'Avg M rat : ', np.mean( Mass_Rat) )


TR_EN = np.array( [ np.sum(im) for im in Truth_En_HighRes ] )
PR_EN = np.array( [ np.sum(im) for im in Pred_En_HighRes ] )

TR_EN = TR_EN[ np.where(TR_EN!=0) ]
PR_EN = PR_EN[ np.where(TR_EN!=0) ]
print('Total PR_EN : ', np.sum(PR_EN) )

print( 'Avg Er : ', np.mean( (PR_EN )/TR_EN ) )
print( 'Sigma Er : ', np.std( (PR_EN)/TR_EN ) )

hf = h5py.File('PredictionFile_SuperRes.h5', 'w')

hf.create_dataset('Total_E_LowRes', data=Total_E_LowRes, compression = "lzf")
hf.create_dataset('Truth_En_HighRes', data=Truth_En_HighRes, compression = "lzf")
hf.create_dataset('Pred_En_HighRes', data=Pred_En_HighRes, compression = "lzf")
hf.create_dataset('Truth_CTheta', data=Truth_Theta, compression = "lzf")
hf.create_dataset('Pred_CTheta', data=Pred_Theta, compression = "lzf")
hf.create_dataset('Mass_Rat_Pi0', data=Mass_Rat, compression = "lzf")

hf.close()  

