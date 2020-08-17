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

nameSourceFile = str(sys.argv[1])
dev_id = str(sys.argv[2])

graph_size = 10

os.environ["CUDA_VISIBLE_DEVICES"]=dev_id


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


# -------------- Define perm equivariant Deepset layer ------------------- #

class PermEqui_mean(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PermEqui_mean, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.Gamma = nn.Sequential(
                          nn.Linear(self.in_dim, 10),
                          nn.LeakyReLU(-0.8),
                          nn.Linear( 10 , 20),
                          nn.LeakyReLU(-0.8),
                          nn.Linear( 20 , 15),
                          nn.LeakyReLU(-0.8),
                          nn.Linear( 15 , 10),
                          nn.LeakyReLU(-0.8),
                          nn.Linear(10, self.out_dim)
                         )

        
        self.Lambda = nn.Sequential(
                          nn.Linear(1, 10),
                          nn.LeakyReLU(-0.8),
                          nn.Linear( 10 , 20),
                          nn.LeakyReLU(-0.8),
                          nn.Linear( 20 , 15),
                          nn.LeakyReLU(-0.8),
                          nn.Linear( 15 , 10),
                          nn.LeakyReLU(-0.8),
                          nn.Linear(10, self.out_dim)
                         )

        
        self.act = nn.Tanh()

    def forward(self, x):
        xm = x.mean(1, keepdim=True)
        xm = self.Lambda(xm) 
        x = self.Gamma(x)
        x = x - xm
        
        return self.act(x)


# ------------ The DeepSet model -------------------- #

class DeepSet(nn.Module):

    def __init__(self, feature_dims, output_classes, input_dims=4):
        super(DeepSet, self).__init__()
        
        self.phi = nn.ModuleList()
        self.num_layers = len(feature_dims)
        
        for i in range(self.num_layers):
            self.phi.append(PermEqui_mean(
                feature_dims[i - 1] if i > 0 else input_dims,
                feature_dims[i])
                            )


        
        self.proj = nn.Sequential(
                    nn.Linear(sum(feature_dims), 20),
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

                    nn.Linear(5, output_classes)
                    )
        
        
    def forward(self, x, x_idx, length):
        
        hs = []
        batch_size, n_points, x_dims = x.shape
        
        h = x

        for i in range(self.num_layers):
            
            h = h.view(batch_size * n_points, -1)
            
            h = self.phi[i](h)
            
            h = F_t.leaky_relu(h, 0.2)
            h = h.view(batch_size, n_points, -1)
            
            
            hs.append(h)
            
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

# --------------- Process a batch element --------------------- #
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


    x_val = (x_val - np.mean(x_val) )/np.std(x_val)
    y_val = (y_val - np.mean(y_val) )/np.std(y_val)
    z_val = (z_val - np.mean(z_val) )/np.std(z_val)

    point_indx = np.array([z_idx, x_idx, y_idx], dtype=int)
    point_indx = np.transpose(point_indx)

    point = np.array([x_val, y_val, z_val, energy_val])
    point = np.transpose(point)  

    point = np.reshape( point, (1, point.shape[0], point.shape[1]) )
    point = torch.FloatTensor(point)


    sample = {
        'Input' : X,
        'seq_length': len(energy_val),
        'point_xyz': point,
        'target': torch.FloatTensor(y),
        'point_idx_zxy' : point_indx,
        'energy' : torch.FloatTensor(energy_val)
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
    
    
    energy = [ torch.FloatTensor(sample['energy']) for sample in batch]
    
    position = [ torch.Tensor(sample['point_xyz']) for sample in batch ]
    position_idx = [ torch.LongTensor(sample['point_idx_zxy']) for sample in batch ]
    
    max_length = np.max(lengths)
    
    n_sequences = len(target)
    
    Input_tensor   = torch.ones((n_sequences,7, 64, 64)).float() * -5
    targets_tensor = torch.ones((n_sequences,6, 64, 64)).float() * -5
    position_idx_tensor = torch.ones((n_sequences,max_length,3)).int() * -1
    position_idx_tensor = position_idx_tensor.long()
    
    position_tensor = torch.ones((n_sequences,max_length,4)) * -999.
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
    
    pos = torch.where(sequence_lengths > graph_size)[0]

    sequence_lengths = sequence_lengths[pos]
    targets_tensor = targets_tensor[pos]
    energy_tensor =  energy_tensor[pos]
    position_tensor = position_tensor[pos]
    position_idx_tensor = position_idx_tensor[pos]
    Input_tensor = Input_tensor[pos]
                            
    return targets_tensor,  sequence_lengths,  energy_tensor, position_tensor, position_idx_tensor,\
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


model = DeepSet( feature_dims = [6, 12, 8, 4], output_classes = 1)
model.to(dev)



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
        for target, length, energy, position, position_idx, Input in tq:
        
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
            

            position_data, target_data = position_data.to(dev), target_data.to(dev)
            Input_data = Input_data.to(dev)

            opt.zero_grad()
            
            output_data = model(position_data, position_idx_data, length)

            del position_idx_data; del position_data; del energy_data; del length;
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
        for target, length, energy, position, position_idx, Input in tq:
        
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
            
            position_data, target_data = position_data.to(dev), target_data.to(dev)
            Input_data = Input_data.to(dev)

            output_data = model(position_data, position_idx_data, length)

            del position_idx_data; del position_data; del energy_data; del length;
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
        torch.save(model.state_dict(), 'model_DeepSet_' + nameSourceFile + '.pt')
        valid_loss_min = valid_loss

