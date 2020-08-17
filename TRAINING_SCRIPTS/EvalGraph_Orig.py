from pathlib import Path

import dgl.function as fn 
from dgl.base import DGLError

from dgl import backend as F
import numpy as np
import os, sys
from scipy import sparse
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
import torch
import dgl
from dgl.utils import expand_as_pair
from dgl.data.utils import download, get_download_dir
from functools import partial
import tqdm

import h5py

from torch import nn, Tensor
import torch.nn.functional as F_t
from torch.nn import init


os.environ["CUDA_VISIBLE_DEVICES"]="0"
cuda_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )


graph_size = 10

nameSourceFile = str(sys.argv[1])

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = '/storage/agrp/sanmay/PFLOW_ANALYSIS/RectangularGeo/CROSS_CHECK/Outputfile_V1_Samples'+nameSourceFile+'.hdf5'

# ------------- code the detector geometry ---------------------- #

X0, L_int = 3.9, 17.4

Z_Val = [ 0, (3*X0/2), (3*X0 + 16*X0/2), (3*X0 + 16*X0 + 6*X0/2), (3*X0 + 16*X0 + 6*X0 + 1 + 1.5*L_int/2),
          (3*X0 + 16*X0 + 6*X0 + 1 + 1.5*L_int + 4.1*L_int/2), 
          (3*X0 + 16*X0 + 6*X0 + 1 + 1.5*L_int + 4.1*L_int + 1.8*L_int/2)
        ]

# -------------- Define granularity ---------- #
Granularity = {
    
    'LayerT' : 64,
    'Layer1' : 64,
    'Layer2' : 32,
    'Layer3' : 32,
    'Layer4' : 16,
    'Layer5' : 16,
    'Layer6' : 8
}


# ----------------- Making the KNN graph ------------------- #
def pairwise_squared_distance(x):
    """
    x : (n_samples, n_points, dims)
    return : (n_samples, n_points, n_points)
    """
    x2s = F.sum(x * x, -1, True)
    # assuming that __matmul__ is always implemented (true for PyTorch, MXNet and Chainer)
    return x2s + F.swapaxes(x2s, -1, -2) - 2 * x @ F.swapaxes(x, -1, -2)

def knn_graph(x, k):
    """Transforms the given point set to a directed graph, whose coordinates
    are given as a matrix. The predecessors of each point are its k-nearest
    neighbors.

    If a 3D tensor is given instead, then each row would be transformed into
    a separate graph.  The graphs will be unioned.

    Parameters
    ----------
    x : Tensor
        The input tensor.

        If 2D, each row of ``x`` corresponds to a node.

        If 3D, a k-NN graph would be constructed for each row.  Then
        the graphs are unioned.
    k : int
        The number of neighbors

    Returns
    -------
    DGLGraph
        The graph.  The node IDs are in the same order as ``x``.
    """
    if F.ndim(x) == 2:
        x = F.unsqueeze(x, 0)
    n_samples, n_points, _ = F.shape(x)

    dist = pairwise_squared_distance(x)
    k_indices = F.argtopk(dist, k, 2, descending=False)
    dst = F.copy_to(k_indices, F.cpu())

    src = F.zeros_like(dst) + F.reshape(F.arange(0, n_points), (1, -1, 1))

    per_sample_offset = F.reshape(F.arange(0, n_samples) * n_points, (-1, 1, 1))
    dst += per_sample_offset
    src += per_sample_offset
    dst = F.reshape(dst, (-1,))
    src = F.reshape(src, (-1,))
    adj = sparse.csr_matrix((F.asnumpy(F.zeros_like(dst) + 1), (F.asnumpy(dst), F.asnumpy(src))))

    g = dgl.DGLGraph(adj,readonly=True)
    return g


class KNNGraph(nn.Module):
    """Layer that transforms one point set into a graph, or a batch of
    point sets with the same number of points into a union of those graphs.

    If a batch of point set is provided, then the point :math:`j` in point
    set :math:`i` is mapped to graph node ID :math:`i \times M + j`, where
    :math:`M` is the number of nodes in each point set.

    The predecessors of each node are the k-nearest neighbors of the
    corresponding point.

    Parameters
    ----------
    k : int
        The number of neighbors
    """
    def __init__(self, k):
        super(KNNGraph, self).__init__()
        self.k = k

    #pylint: disable=invalid-name
    def forward(self, x):
        """Forward computation.

        Parameters
        ----------
        x : Tensor
            :math:`(M, D)` or :math:`(N, M, D)` where :math:`N` means the
            number of point sets, :math:`M` means the number of points in
            each point set, and :math:`D` means the size of features.

        Returns
        -------
        DGLGraph
            A DGLGraph with no features.
        """
        return knn_graph(x, self.k)


# ---------------- The EdgeConv function ---------------------- #
class EdgeConv(nn.Module):
    r"""EdgeConv layer.

    Introduced in "`Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__".  Can be described as follows:

    .. math::
       x_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \mathrm{ReLU}(
       \Theta \cdot (x_j^{(l)} - x_i^{(l)}) + \Phi \cdot x_i^{(l)})

    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    batch_norm : bool
        Whether to include batch normalization on messages.
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 batch_norm=False):
        super(EdgeConv, self).__init__()
        self.batch_norm = batch_norm
        
        self.theta_en = nn.Sequential(
        nn.Linear(1, 64),
        #nn.BatchNorm1d(64),
        nn.LeakyReLU( -0.5 ), 
        nn.Linear(64, 128),
        #nn.BatchNorm1d(128),
        nn.LeakyReLU( -0.5 ),
        nn.Linear(128, 32),
        #nn.BatchNorm1d(32),
        nn.LeakyReLU( -0.5 ),
        nn.Linear(32, 1)
        )
        
        self.phi_en = nn.Sequential(
        nn.Linear(1, 64),
        #nn.BatchNorm1d(64),
        nn.LeakyReLU( -0.5 ), 
        nn.Linear(64, 128),
        #nn.BatchNorm1d(128),
        nn.LeakyReLU( -0.5 ),
        nn.Linear(128, 32),
        #nn.BatchNorm1d(32),
        nn.LeakyReLU( -0.5 ),
        nn.Linear(32, 1)
        )

        self.theta = nn.Linear(in_feat, out_feat)
        self.phi = nn.Linear(in_feat, out_feat)

        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feat)

    def message(self, edges):
        """The message computation function.
        """
        theta_x = self.theta(edges.dst['x'] - edges.src['x'])
        phi_x = self.phi(edges.src['x'])
        
        theta_en = self.theta_en(edges.dst['en'] - edges.src['en'])
        phi_en = self.phi_en(edges.src['en'])
        
        return {'e': theta_x + phi_x, 
                'e_en' :  phi_en + theta_en 
                }

    def forward(self, g, h, h_en):
        """Forward computation

        """
        with g.local_scope():
            h_src, h_dst = expand_as_pair(h)
            h_src_en, h_dst_en = expand_as_pair(h_en)
            
            g.srcdata['x'] = h_src
            g.dstdata['x'] = h_dst
            
            g.srcdata['en'] = h_src_en
            g.dstdata['en'] = h_dst_en
            
            if not self.batch_norm:
                #g.update_all(self.message, fn.mean('e', 'x'))
                g.apply_edges(self.message)
                g.update_all(fn.copy_e('e', 'e'), fn.max('e', 'x'))
                g.update_all(fn.copy_e('e_en', 'e_en'), fn.mean('e_en', 'en'))
            else:
                g.apply_edges(self.message)

                g.edata['e'] = self.bn(g.edata['e'])
                
                g.update_all(fn.copy_e('e', 'e'), fn.max('e', 'x'))
                 
                g.update_all(fn.copy_e('e_en', 'e_en'), fn.mean('e_en', 'en'))
                
            return g.dstdata['x'], g.dstdata['en'] #+  h_en 



# ------------------- The final Model function ------------------- #
class Model(nn.Module):
    def __init__(self, k, feature_dims, output_classes, input_dims=3,
                 dropout_prob=0.5):
        super(Model, self).__init__()

        self.nng = KNNGraph(k)
        self.conv = nn.ModuleList()

        self.num_layers = len(feature_dims)
        for i in range(self.num_layers):
            self.conv.append(EdgeConv(
                feature_dims[i - 1] if i > 0 else input_dims,
                feature_dims[i],
                batch_norm=True))

        
        self.proj = nn.Sequential(
                    nn.Linear(len(feature_dims), 20),
                    nn.LeakyReLU( -0.8 ), 

                    nn.Linear(20, 40),                   
                    nn.LeakyReLU( -0.8 ), 

                    nn.Linear(40, 30),                   
                    nn.LeakyReLU( -0.8 ),

                    nn.Linear(30, 10),                   
                    nn.LeakyReLU( -0.8 ),

                    nn.Linear(10, 5),                    
                    nn.LeakyReLU( -0.8 ),

                    nn.Linear(5, output_classes)
                    )

    

    def forward(self, gr, x_idx, x, x_en, length):
        hs = []
        batch_size, n_points, x_dims = x.shape
        h, h_en = x, x_en

        for i in range(self.num_layers):
            
            if(i == 0) : g = gr
            else : 
                g = self.nng(h)
                g = dgl.transform.remove_self_loop(g)
                
            h = h.view(batch_size * n_points, -1)
            h_en = h_en.view(batch_size * n_points, -1)
            h, h_en = self.conv[i](g, h, h_en)
            
            h = F_t.leaky_relu(h, 0.2)
            h = h.view(batch_size, n_points, -1)
            
            h_en = h_en.view(batch_size, n_points, -1)
            
            hs.append(h_en)

        h = torch.cat(hs, 2)
        h = h.view(batch_size * n_points, -1)
        h = self.proj(h)
        
        out_image = []
        
        for i_seq in range( len(length) ) : 
            
            e_seq = h[ torch.sum(length[:i_seq]) :  torch.sum(length[:i_seq]) + length[i_seq] ]
            e_seq = torch.reshape( e_seq, (e_seq.shape[0],) )
            idx_seq = x_idx[ torch.sum(length[:i_seq]) :  torch.sum(length[:i_seq]) + length[i_seq] ]
            
            img = torch.zeros([7, 64, 64], device=dev)
            
            img[ idx_seq[:,0], idx_seq[:,1], idx_seq[:,2] ] = e_seq
            
            img = torch.reshape(img, (1, 7, 64, 64))
            out_image.append(img)

        #h = torch.tanh(h)
        #h = torch.clamp(h, min=-1., max=1.) 
        
        out_image = torch.cat(out_image, axis=0)

        return out_image[:, 1:, :, :]


# ----------------- Function for processing each layer ------------------ #
def MakeLayer(layer, layer_tar, index) :
    
    if(index == 6) : str_layer = 'Layer' + 'T'
    else :           str_layer = 'Layer' + str(index+1)
        
    gran  =  Granularity[str_layer]
    
    layer = layer[0 : gran, 0 : gran]
    
    
    loc = np.where(layer!=0)
    loc_np = np.array(loc)
    loc_np = np.transpose(loc_np)
    
    #print(loc)
    
    layer_idx = -1
    
    if(index == 6) : layer_idx = 0
    else : layer_idx = index+1

    x_val = loc[0] / float(gran) * 125 - 125./2
    y_val = loc[1] / float(gran) * 125 - 125./2
    z_val = Z_Val[layer_idx] * np.ones( x_val.shape )
    en_val  = layer[loc]
    en_tar  = layer_tar[loc]
    
    
    point = []
    point.append( en_val  )
    #point.append( layer_idx * np.ones( en_val.shape )  )
    point.append( x_val )
    point.append( y_val )
    point.append( z_val )
    
    point.append( loc[0]  )
    point.append( loc[1]  )
    point.append( layer_idx * np.ones( en_val.shape ) )
    
    point.append( en_tar  )
    
    point = np.array( point )
    
    
    return point


def ProcessImage(batch_itr) : 
    
    X, y = batch_itr
    
    ybar = np.ones([64, 64]) * -2.
    Y = np.stack([y[0], y[1], y[2], y[3], y[4], y[5], ybar], axis=0)
    
    test1, test2, test3, test4, test5, test6, testT =\
    MakeLayer(X[0], Y[0], 0), MakeLayer(X[1], Y[1], 1), \
    MakeLayer(X[2], Y[2], 2), MakeLayer(X[3], Y[3], 3), \
    MakeLayer(X[4], Y[4], 4), MakeLayer(X[5], Y[5], 5), \
    MakeLayer(X[6], Y[6], 6)



    energy_val = np.concatenate([ testT[0], test1[0], test2[0], test3[0], test4[0], test5[0], test6[0] ])
    x_val   = np.concatenate([ testT[1], test1[1], test2[1], test3[1], test4[1], test5[1], test6[1] ])
    y_val   = np.concatenate([ testT[2], test1[2], test2[2], test3[2], test4[2], test5[2], test6[2] ])
    z_val   = np.concatenate([ testT[3], test1[3], test2[3], test3[3], test4[3], test5[3], test6[3] ])
    x_idx   = np.concatenate([ testT[4], test1[4], test2[4], test3[4], test4[4], test5[4], test6[4] ])
    y_idx   = np.concatenate([ testT[5], test1[5], test2[5], test3[5], test4[5], test5[5], test6[5] ])
    z_idx   =np.concatenate([ testT[6], test1[6], test2[6], test3[6], test4[6], test5[6], test6[6] ])
    target_val = np.concatenate([ testT[7], test1[7], test2[7], test3[7], test4[7], test5[7], test6[7] ])



    point_indx = np.array([z_idx, x_idx, y_idx], dtype=int)
    point_indx = np.transpose(point_indx)

    point = np.array([x_val, y_val, z_val])
    point = np.transpose(point)  

    point = np.reshape( point, (1, point.shape[0], point.shape[1]) )
    point = torch.FloatTensor(point)

#         x_red = x[loc]
#         y_red = y[loc]

    graph = KNNGraph(graph_size)

    npoints = energy_val.shape[0]
    
    if(npoints < graph_size) :         
        g = dgl.DGLGraph()
        g.add_nodes(2)
    else : 
        g = graph(point)
        g = dgl.transform.remove_self_loop(g)

    sample = {
        'Input' : X,
        'seq_length': len(energy_val),
        'point_xyz': point,
        'target': torch.FloatTensor(y),
        'point_idx_zxy' : point_indx,
        'energy' : torch.FloatTensor(energy_val),
        'gr' : g
    }

    return sample

# ------ the custom batch function ----- #
def create_batch(batch_all):
    
    #print('Batch size : ', len(batch_all))
    
    #X, Y = batch_all[0]
    
    batch = [ ProcessImage(batch_itr) for batch_itr in batch_all ]
    
    lengths = [sample['seq_length'] for sample in batch]
        
    Input  = [ torch.FloatTensor(sample['Input']) for sample in batch]    
    target = [ torch.FloatTensor(sample['target']) for sample in batch]
    #position = [ torch.FloatTensor(sample['position']) for sample in batch  ]
    graph = [ sample['gr'] for sample in batch ]
    
    energy = [ torch.FloatTensor(sample['energy']) for sample in batch]
    
    position = [ torch.Tensor(sample['point_xyz']) for sample in batch ]
    position_idx = [ torch.LongTensor(sample['point_idx_zxy']) for sample in batch ]
    
    max_length = np.max(lengths)
    
    n_sequences = len(target)
    
    Input_tensor   = torch.ones((n_sequences,7, 64, 64)).float() * -5
    targets_tensor = torch.ones((n_sequences,6, 64, 64)).float() * -5
    position_idx_tensor = torch.ones((n_sequences,max_length,3)).int() * -1
    position_idx_tensor = position_idx_tensor.long()
    
    position_tensor = torch.ones((n_sequences,max_length,3)).int() * -999.
    energy_tensor = torch.zeros((n_sequences,max_length)).float()
    
    
    for i in range(n_sequences):
        seq_len = lengths[i]
        
        Input_tensor[i]   = Input[i]
        targets_tensor[i] = target[i]
        position_tensor[i,:seq_len] = position[i]
        position_idx_tensor[i,:seq_len] = position_idx[i]
        energy_tensor[i,:seq_len] = energy[i]
        
    
    sequence_lengths = torch.LongTensor(lengths)
    
    sequence_lengths, idx = sequence_lengths.sort(dim=0, descending=True)
    
    targets_tensor = targets_tensor[idx]
    energy_tensor =  energy_tensor[idx]
    position_tensor = position_tensor[idx]
    position_idx_tensor = position_idx_tensor[idx]
    Input_tensor = Input_tensor[idx]
    graph = sorted(graph, key=lambda g: g.number_of_nodes(), reverse=True)
    
    pos = torch.where(sequence_lengths > graph_size)[0] 
    
    sequence_lengths = sequence_lengths[pos]
    targets_tensor = targets_tensor[pos]
    energy_tensor =  energy_tensor[pos]
    position_tensor = position_tensor[pos]
    position_idx_tensor = position_idx_tensor[pos]
    Input_tensor = Input_tensor[pos]
    graph = graph[0: len(pos) ]
                            
    return dgl.batch(graph),  targets_tensor,  sequence_lengths,  energy_tensor, position_tensor, position_idx_tensor,\
           Input_tensor 
    


# ------ read the dataset -------- #
ntrain, nvalid = -1, -1
f = h5py.File(file_path, 'r')

test_data_input = f['test']['input'][:ntrain]
test_data_target = f['test']['output'][:ntrain]


test_dataset = TensorDataset( Tensor(  torch.from_numpy(test_data_input).float() ), Tensor( torch.from_numpy(test_data_target).float() ) )

del test_data_input; del test_data_target;

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=5, shuffle=False,collate_fn=create_batch)

model = Model(graph_size, feature_dims=[32, 64,  128, 64, 3], output_classes=1)
model = model.to(dev)


model.load_state_dict(torch.load('model_Graph_'+nameSourceFile+'.pt'))
model.eval()


Total_E, Target_Fr, Output_Fr = [], [], []


with tqdm.tqdm(test_loader, ascii=True) as tq:
    for gr ,target, length, energy, position, position_idx, Input in tq:
    
        position_data, position_idx_data, target_data, energy_data, Input_data = [], [], [], [], []

        it = 0
        for i_nd in length : 
            position_data.append( position[it][0:i_nd] )
            position_idx_data.append( position_idx[it][0:i_nd] )
            target_data.append( torch.reshape(target[it][0:i_nd], (1, target[it][0:i_nd].shape[0], target[it][0:i_nd].shape[1], target[it][0:i_nd].shape[2])) )
            energy_data.append( energy[it][0:i_nd] )
            Input_data.append( torch.reshape( Input[it][0:i_nd], (1, Input[it][0:i_nd].shape[0], Input[it][0:i_nd].shape[1], Input[it][0:i_nd].shape[2]) ) )
            it = it+1

        target_data = torch.cat(target_data, axis=0)
        Input_data = torch.cat(Input_data, axis=0)
        energy_data = torch.cat(energy_data, axis=0)

        position_data = torch.cat(position_data, axis=0)
        position_data = torch.reshape(position_data, (1, position_data.shape[0], position_data.shape[1]) )
        
        position_idx_data = torch.cat(position_idx_data, axis=0)
        
        gr, position_data, target_data, energy_data = gr.to(dev), position_data.to(dev), target_data.to(dev), energy_data.to(dev)
        Input_data = Input_data.to(dev)

        output_data = model(gr, position_idx_data, position_data, energy_data, length)

        del gr; del position_idx_data; del position_data; del energy_data; del length;
        torch.cuda.empty_cache()

        Total_E.append( Input_data.cpu().detach().numpy() )
        Target_Fr.append( target_data.cpu().detach().numpy() )
        Output_Fr.append( output_data.cpu().detach().numpy() )


Total_E, Target_Fr, Output_Fr = np.concatenate(Total_E), np.concatenate(Target_Fr), np.concatenate(Output_Fr)


hf = h5py.File('PredictionFile_Graph_' + nameSourceFile + '_V1.h5', 'w')

hf.create_dataset('Total_E', data=Total_E)
hf.create_dataset('Target_Fr', data=Target_Fr)
hf.create_dataset('Output_Fr', data=Output_Fr)

hf.close()        
