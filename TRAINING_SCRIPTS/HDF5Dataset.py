import os
import numpy as np
import h5py
from torch.utils.data import TensorDataset
from pathlib import Path
import torch



class HDF5Dataset(TensorDataset):
    """Represents an abstract HDF5 dataset.
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, file_path, TypeData, recursive):
        super().__init__()
        self.file_path = file_path 
        # self.data_cache_size = data_cache_size
        # self.transform = transform
        self.TypeData = TypeData
        
        # Search for all h5 files
        p = Path(file_path)
        # assert(p.is_dir())
        # if recursive:
            # files = sorted(p.glob('**/*.hdf5'))
        # else:
            # files = sorted(p.glob('*NegPosTarget.hdf5'))
        self.file_path = [p]
        # if len(files) < 1:
        #     raise RuntimeError('No hdf5 datasets found')
        pp = str(self.file_path[0])
        self.op = h5py.File( pp,'r')

            
    def __getitem__(self, index):
        # pp = str(self.file_path[0])
        # op = h5py.File( pp,'r')
        x = self.op[self.TypeData]["input"][index]#self.get_data("input",index,op)
        x = torch.from_numpy(x).float()
        y = self.op[self.TypeData]["output"][index]#self.get_data("output", index,op)
        y = torch.from_numpy(y).float()
        # print ("index",index)
        # op.close()
        return (x, y)

    def __len__(self):
        # pp = str(self.file_path[0])
        # self.op = h5py.File( pp,'r')
        len_ = self.op[self.TypeData]['input'].shape[0]
        # op.close()
        # print ("len",len_)
        return int(len_)

    def close_ (self):
        self.op.close()

