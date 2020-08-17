import os
import sys
import random

import numpy as np
import pandas as pd

import h5py

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain

import math

import uproot

import tools_generic


f = uproot.open('/storage/agrp/jshlomi/pflow/PFlowNtupleFile_HOMDet_' + sys.argv[1] + 'GeV_Overlap_WS.root')
nameFile = 'Outputfile_V1_Samples' + str(sys.argv[1])



print(f['EventTree'].keys())

NStart = 100000
NEvent = 6000

Trk_X_pos = f['EventTree'].array('Trk_X_pos', entrystart = NStart , entrystop = NStart + NEvent  )
Trk_Y_pos = f['EventTree'].array('Trk_Y_pos', entrystart = NStart , entrystop = NStart + NEvent  )

Trk_Theta = f['EventTree'].array('Trk_Theta', entrystart = NStart , entrystop = NStart + NEvent  )
Trk_Phi   = f['EventTree'].array('Trk_Phi', entrystart = NStart , entrystop = NStart + NEvent  )

hfile = h5py.File('/storage/agrp/sanmay/PFLOW_ANALYSIS/RectangularGeo/CROSS_CHECK/TOPO_FILES/TrkInfo_'+nameFile+'.hdf5','w')

hfile.create_dataset('Trk_X_pos', data=Trk_X_pos, compression = "lzf")
hfile.create_dataset('Trk_Y_pos', data=Trk_Y_pos, compression = "lzf")
hfile.create_dataset('Trk_Theta', data=Trk_Theta, compression = "lzf")
hfile.create_dataset('Trk_Phi', data=Trk_Phi, compression = "lzf")
hfile.close()



