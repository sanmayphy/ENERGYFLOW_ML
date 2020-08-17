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
from dgl.nn.pytorch.utils import Identity
from dgl.data.utils import download, get_download_dir
from functools import partial
import tqdm

import h5py

import torch as th

from torch import nn, Tensor
import torch.nn.functional as F_t
from torch.nn import init


os.environ["CUDA_VISIBLE_DEVICES"]="0"
cuda_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )

graph_size = 10

nameSourceFile = str(sys.argv[1])

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = '/storage/agrp/sanmay/PFLOW_ANALYSIS/RectangularGeo/CROSS_CHECK/Outputfile_'+nameSourceFile+'.hdf5'


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


# ------------ the GMM layer --------------------- #
class GMMConv(nn.Module):
    r"""The Gaussian Mixture Model Convolution layer from `Geometric Deep
    Learning on Graphs and Manifolds using Mixture Model CNNs
    <http://openaccess.thecvf.com/content_cvpr_2017/papers/Monti_Geometric_Deep_Learning_CVPR_2017_paper.pdf>`__.

    .. math::
        h_i^{l+1} & = \mathrm{aggregate}\left(\left\{\frac{1}{K}
         \sum_{k}^{K} w_k(u_{ij}), \forall j\in \mathcal{N}(i)\right\}\right)

        w_k(u) & = \exp\left(-\frac{1}{2}(u-\mu_k)^T \Sigma_k^{-1} (u - \mu_k)\right)

    Parameters
    ----------
    in_feats : int
        Number of input features.
    out_feats : int
        Number of output features.
    dim : int
        Dimensionality of pseudo-coordinte.
    n_kernels : int
        Number of kernels :math:`K`.
    aggregator_type : str
        Aggregator type (``sum``, ``mean``, ``max``).
    residual : bool
        If True, use residual connection inside this layer. Default: ``False``.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 dim,
                 n_kernels,
                 aggregator_type='sum',
                 residual=False,
                 bias=True):
        super(GMMConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._dim = dim
        self._n_kernels = n_kernels
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        elif aggregator_type == 'max':
            self._reducer = fn.max
        else:
            raise KeyError("Aggregator type {} not recognized.".format(aggregator_type))
            
        self._reducer_e = fn.mean

        self.mu = nn.Parameter(th.Tensor(n_kernels, dim))
        self.inv_sigma = nn.Parameter(th.Tensor(n_kernels, dim))
        #self.fc = nn.Linear(self._in_src_feats, n_kernels * out_feats, bias=False)

        self.fc = nn.Sequential(
                        nn.Linear(self._in_src_feats, 20),
                        nn.LeakyReLU( -0.8 ),
                        nn.Linear(20, 40),
                        nn.LeakyReLU( -0.8 ),
                        nn.Linear(40, 60),
                        nn.LeakyReLU( -0.8 ),
                        nn.Linear(60, n_kernels * out_feats),
                        nn.LeakyReLU( -0.8 ),
        	)        
        
        self.mu_e = nn.Parameter(th.Tensor(n_kernels, dim))
        self.inv_sigma_e = nn.Parameter(th.Tensor(n_kernels, dim))
        #self.fc_e = nn.Linear(1, n_kernels , bias=False)

        self.fc_e = nn.Sequential(
                        nn.Linear(1, 20),
                        nn.LeakyReLU( -0.8 ),
                        nn.Linear(20, 40),
                        nn.LeakyReLU( -0.8 ),
                        nn.Linear(40, 60),
                        nn.LeakyReLU( -0.8 ),
                        nn.Linear(60, n_kernels),
                        nn.LeakyReLU( -0.8 ),
        	)
        
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))           
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = init.calculate_gain('relu')
        # init.xavier_normal_(self.fc.weight, gain=gain)
        
        # init.xavier_normal_(self.fc_e.weight, gain=gain)
        
        if isinstance(self.res_fc, nn.Linear):
            init.xavier_normal_(self.res_fc.weight, gain=gain)
        init.normal_(self.mu.data, 0, 0.1)
        init.constant_(self.inv_sigma.data, 1)
        
        init.normal_(self.mu_e.data, 0, 0.1)
        init.constant_(self.inv_sigma_e.data, 1)
        
        if self.bias is not None:
            init.zeros_(self.bias.data)

    def forward(self, graph, feat, feat_e, pseudo):
        """Compute Gaussian Mixture Model Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            If a single tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of tensors are given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        pseudo : torch.Tensor
            The pseudo coordinate tensor of shape :math:`(E, D_{u})` where
            :math:`E` is the number of edges of the graph and :math:`D_{u}`
            is the dimensionality of pseudo coordinate.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is the output feature size.
        """
      
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat)
            graph.srcdata['h'] = self.fc(feat_src).view(-1, self._n_kernels, self._out_feats)
            
            feat_src_e, feat_dst_e = expand_as_pair(feat_e)
            graph.srcdata['h_e'] = self.fc_e(feat_src_e).view(-1, self._n_kernels, 1)
            
            E = graph.number_of_edges()
            # compute gaussian weight
            gaussian = -0.5 * ((pseudo.view(E, 1, self._dim) -
                                self.mu.view(1, self._n_kernels, self._dim)) ** 2)
            gaussian = gaussian * (self.inv_sigma.view(1, self._n_kernels, self._dim) ** 2)
            gaussian = th.exp(gaussian.sum(dim=-1, keepdim=True)) # (E, K, 1)
            graph.edata['w'] = gaussian
            graph.update_all(fn.u_mul_e('h', 'w', 'm'), self._reducer('m', 'h'))
            rst = graph.dstdata['h'].sum(1)
            
            gaussian_e = -0.5 * ((pseudo.view(E, 1, self._dim) -
                                self.mu_e.view(1, self._n_kernels, self._dim)) ** 2)
            gaussian_e = gaussian_e * (self.inv_sigma_e.view(1, self._n_kernels, self._dim) ** 2)
            gaussian_e = th.exp(gaussian_e.sum(dim=-1, keepdim=True)) # (E, K, 1)
            graph.edata['w_e'] = gaussian_e
            graph.update_all(fn.u_mul_e('h_e', 'w_e', 'm_e'), self._reducer_e('m_e', 'h_e'))
            rst_e = graph.dstdata['h_e'].sum(1)
            
            # residual connection
            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)
            # bias
            if self.bias is not None:
                rst = rst + self.bias
            return rst, rst_e



# ------------------- The final Model function ------------------- #
class Model(nn.Module):
    def __init__(self, k, feature_dims, kernel_dims, output_classes, input_dims=3):
        super(Model, self).__init__()

        self.nng = KNNGraph(k)
        self.conv = nn.ModuleList()

        self.num_layers = len(feature_dims)
        for i in range(self.num_layers):
            self.conv.append(GMMConv(
                in_feats = (feature_dims[i - 1] if i > 0 else input_dims),
                out_feats =feature_dims[i],
                dim = 4,
                n_kernels = kernel_dims[i])
                            )

        
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
            
            #g = gr

            pseudo = []
            us, vs = g.edges()
            for ie in range(g.number_of_edges()):
                pseudo.append([
                    1 / np.sqrt(g.in_degree(us[ie])),
                    1 / np.sqrt(g.in_degree(vs[ie])),
                    1 / np.sqrt(g.in_degree(us[ie])),
                    1 / np.sqrt(g.in_degree(vs[ie]))
                ])
            pseudo = torch.Tensor(pseudo)
            pseudo = pseudo.to(dev)
            
            h = h.view(batch_size * n_points, -1)
            h_en = h_en.view(batch_size * n_points, -1)
            
            h, h_en = self.conv[i](g, h, h_en, pseudo)
            
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

train_data_input = f['train']['input'][:ntrain]
train_data_target = f['train']['output'][:ntrain]

valid_data_input = f['valid']['input'][:nvalid]
valid_data_target = f['valid']['output'][:nvalid]

train_dataset = TensorDataset( Tensor(  torch.from_numpy(train_data_input).float() ), Tensor( torch.from_numpy(train_data_target).float() ) )
valid_dataset = TensorDataset( Tensor(  torch.from_numpy(valid_data_input).float() ),   Tensor( torch.from_numpy(valid_data_target).float()   ) )


del train_data_input; del train_data_target; 
del valid_data_input; del valid_data_target;


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True,collate_fn=create_batch)

valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=5, shuffle=True,collate_fn=create_batch)




model = Model(graph_size, feature_dims=[32, 64,  128, 64, 3], kernel_dims=[20, 30, 60, 40, 10], output_classes=1)
model = model.to(dev)

# ---- the loss function ---------- #
def LossFunction(pred, tar, total) : 
    

        
    pos = torch.where( total != 0 )
    
    total = total[:, 0:6, :, :]
    
    loc = torch.where( total >= 0. )
    total = total[loc]
    pred = pred[loc]
    tar = tar[loc]
    
#     target_energy = tar * total

    wt_avg = torch.sum( total.to(dev) * torch.abs( pred.to(dev) - tar.to(dev) ) **2  )
    #print('Before div : ', wt_avg)
    wt_avg = wt_avg / torch.sum( total.to(dev)  )
    
    return wt_avg

opt = optim.Adam(model.parameters(), lr=1e-4)


# ---------------- Make the training loop ----------------- #

train_loss_v, valid_loss_v = [], []


# number of epochs to train the model
n_epochs = 50

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

            opt.zero_grad()
            
            output_data = model(gr, position_idx_data, position_data, energy_data, length)

            del gr; del position_idx_data; del position_data; del energy_data; del length;
            torch.cuda.empty_cache()
            
            #print('Output.shape : ', output_data.shape )
        
            loss = LossFunction(output_data, target_data, Input_data)
            
            loss.backward()
            # perform a single optimization step (parameter update)
            opt.step()

            # update training loss
            train_loss += loss.item() * target_data.shape[0]
        

        
    ######################  
    
    
    
    ######################    
    # validate the model #
    ######################
    model.eval()
    with tqdm.tqdm(valid_loader, ascii=True) as tq:
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
        
            loss = LossFunction(output_data, target_data, Input_data)

            # update training loss
            valid_loss += loss.item() * target_data.shape[0]
    
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
        torch.save(model.state_dict(), 'model_GMM_' + nameSourceFile + '.pt')
        valid_loss_min = valid_loss
