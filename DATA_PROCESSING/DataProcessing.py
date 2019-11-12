import os
import sys
import random

import numpy as np
import pandas as pd

import h5py
import argparse

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain

import math

import uproot 

from GlobalFunctions import MakeTrackLayer, MakeRealResolution, MakeFraction, MakeEnergyImage, \
     MakeTruthTrajectory

parser = argparse.ArgumentParser()
parser.add_argument("--nevent", help="total number of events in the file",
                    type=int, default=3000)
parser.add_argument("--input_file", help="input file name",
                    type=str, default='../PFlowNtupleFile_DiffPos_AllInfo.root')


args = parser.parse_args()

NEvent = args.nevent
fileName = args.input_file

f = uproot.open(fileName)

print(f['EventTree'].keys() )

entrystart = 0 # the test image indices will start from this number. warning!! hard coded

# ------------ read the calorimeter + track variables from the input file --------- #
Layer = f['EventTree'].array('cell_Energy', entrystart = entrystart, entrystop = entrystart+NEvent) # -- total -- #
Ch_Layer = f['EventTree'].array('cellCh_Energy', entrystart = entrystart, entrystop = entrystart+NEvent)
Nu_Layer = f['EventTree'].array('cellNu_Energy', entrystart = entrystart, entrystop = entrystart+NEvent)
Noise_Layer = f['EventTree'].array('Noise_cell_Energy', entrystart = entrystart, entrystop = entrystart+NEvent)

Trk_X_pos = f['EventTree'].array('Trk_X_pos', entrystart = entrystart, entrystop = entrystart+NEvent)
Trk_Y_pos = f['EventTree'].array('Trk_Y_pos', entrystart = entrystart, entrystop = entrystart+NEvent)

Trk_X_indx = f['EventTree'].array('Trk_X_indx', entrystart = entrystart, entrystop = entrystart+NEvent)
Trk_Y_indx = f['EventTree'].array('Trk_Y_indx', entrystart = entrystart, entrystop = entrystart+NEvent)

Track_Energy = f['EventTree'].array('Smeared_Ch_Energy', entrystart = entrystart, entrystop = entrystart+NEvent)
Trk_Theta = f['EventTree'].array('Trk_Theta', entrystart = entrystart, entrystop = entrystart+NEvent)
Trk_Phi   = f['EventTree'].array('Trk_Phi'  , entrystart = entrystart, entrystop = entrystart+NEvent)

Pi0_Theta = f['EventTree'].array('Pi0_Theta', entrystart = entrystart, entrystop = entrystart+NEvent)
Pi0_Phi = f['EventTree'].array('Pi0_Phi', entrystart = entrystart, entrystop = entrystart+NEvent)

Photon1_E = f['EventTree'].array('Photon1_E', entrystart = entrystart, entrystop = entrystart+NEvent)
Photon1_Theta = f['EventTree'].array('Photon1_Theta', entrystart = entrystart, entrystop = entrystart+NEvent)
Photon1_Phi = f['EventTree'].array('Photon1_Phi', entrystart = entrystart, entrystop = entrystart+NEvent)

Photon2_E = f['EventTree'].array('Photon2_E', entrystart = entrystart, entrystop = entrystart+NEvent)
Photon2_Theta = f['EventTree'].array('Photon2_Theta', entrystart = entrystart, entrystop = entrystart+NEvent)
Photon2_Phi = f['EventTree'].array('Photon2_Phi', entrystart = entrystart, entrystop = entrystart+NEvent)

# -------- add noise to the original total energy --------- #
Orig_Layer = Layer # --- renaming the original energy tensor ----- #
Layer = Layer + Noise_Layer

# ------ make the images for track layer ------- #
Track_Layer = np.array( [ MakeTrackLayer(Track_Energy[it], Trk_X_pos[it], Trk_Y_pos[it]) for  it in range(len(Track_Energy))] )

# ------ make the real resolution for total energy ------------- #
RealRes = np.array([ MakeRealResolution(Layer[i_img], Track_Layer[i_img]) for i_img in range( len(Layer) )  ])

# ------ make the real resolution for original energy ------------- #
Orig_RealRes = np.array([ MakeRealResolution(Orig_Layer[i_img], Track_Layer[i_img]) for i_img in range( len(Orig_Layer) )  ])

# ------ make the real resolution for truth neutral energy ------------- #
truth_Nu_RealRes = np.array([ MakeRealResolution(Nu_Layer[i_img], Track_Layer[i_img]) for i_img in range( len(Nu_Layer) )  ])

# ------ make the real resolution for truth charged energy ------------- #
truth_Ch_RealRes = np.array([ MakeRealResolution(Ch_Layer[i_img], Track_Layer[i_img]) for i_img in range( len(Ch_Layer) )  ])

# ------ make the truth extrapolated layer for Pi+ --------- #
ChargePi_CellIndex_Layer = np.array( [MakeTruthTrajectory(Trk_Theta[ievt], Trk_Phi[ievt], Trk_X_indx[ievt], Trk_X_indx[ievt]) for ievt in range( NEvent )  ] )

# ------ make the truth extrapolated layer for Pi0 --------- #
NeutralPi_CellIndex_Layer = np.array( [MakeTruthTrajectory(Pi0_Theta[ievt], Pi0_Phi[ievt], Trk_X_indx[ievt], Trk_X_indx[ievt]) for ievt in range( NEvent )  ] )

# ----- create the co-ordinates of cell mid-points --- #
Layer_Pixel = [32, 64, 32, 16, 16, 8] # --> granularity of the six layers --- #

layer_cell_x = []
for layer_i in range(6) : 
    layer_cell_x.append( np.array([ -125. + (125./Layer_Pixel[layer_i]) + i_x * 250./Layer_Pixel[layer_i]  for i_x in range(Layer_Pixel[layer_i]) ]) )


# ----- hard coding the layer depth for z direction -------- #
X0_ECAL = 0.5 + 14.0
Lambda_int = 16.8 + 79.4

Total_ECAL_Length = 3 * X0_ECAL + 16 * X0_ECAL + 6 * X0_ECAL
Total_HCAL_Length = 1.5 * Lambda_int + 4.1 * Lambda_int + 1.8 * Lambda_int
Total_Calo_Length = Total_ECAL_Length + Total_HCAL_Length + 1.0 # -- there is a 1 cm gap between ECAL & HCAL


zpos_ECAL1 = -1 * Total_Calo_Length/2 + 3 * X0_ECAL/2
zpos_ECAL2 = -1 * Total_Calo_Length/2 + 3 * X0_ECAL + 16 * X0_ECAL/2
zpos_ECAL3 = -1 * Total_Calo_Length/2 + 3 * X0_ECAL + 16 * X0_ECAL + 6 * X0_ECAL/2

zpos_HCAL1 = -1 * Total_Calo_Length/2 + 3 * X0_ECAL + 16 * X0_ECAL + 6 * X0_ECAL + 1 + 1.5 * Lambda_int/2
zpos_HCAL2 = -1 * Total_Calo_Length/2 + 3 * X0_ECAL + 16 * X0_ECAL + 6 * X0_ECAL + 1 + 1.5 * Lambda_int + 4.1 * Lambda_int/2
zpos_HCAL3 = -1 * Total_Calo_Length/2 + 3 * X0_ECAL + 16 * X0_ECAL + 6 * X0_ECAL + 1 + 1.5 * Lambda_int + 4.1 * Lambda_int + 1.8 * Lambda_int/2


zpos_Arr = np.array([zpos_ECAL1, zpos_ECAL2, zpos_ECAL3, zpos_HCAL1, zpos_HCAL2, zpos_HCAL3])


with h5py.File('Outfile_CellInformation.h5', 'w') as f:
    for layer_i in range(6) : 
        f.create_dataset('UniformRes_TotalEnergy_Layer' + str(layer_i+1), data=Layer[:, layer_i:layer_i+1, :, :])
        #f.create_dataset('UniformRes_NoiseEnergy_Layer' + str(layer_i+1), data=Noise_Layer[:5, layer_i:layer_i+1, :, :])
        f.create_dataset('UniformRes_ChargedEnergy_Layer' + str(layer_i+1), data=Ch_Layer[:, layer_i:layer_i+1, :, :])
        f.create_dataset('UniformRes_NeutralEnergy_Layer' + str(layer_i+1), data=Nu_Layer[:, layer_i:layer_i+1, :, :])
        f.create_dataset('RealRes_TotalEnergy_Layer' + str(layer_i+1), data=RealRes[ :, layer_i:layer_i+1,  0:Layer_Pixel[layer_i], 0:Layer_Pixel[layer_i] ] )
        f.create_dataset('RealRes_ChargedEnergy_Layer' + str(layer_i+1), data=truth_Ch_RealRes[ :, layer_i:layer_i+1,  0:Layer_Pixel[layer_i], 0:Layer_Pixel[layer_i] ] )
        f.create_dataset('RealRes_NeutralEnergy_Layer' + str(layer_i+1), data=truth_Nu_RealRes[ :, layer_i:layer_i+1,  0:Layer_Pixel[layer_i], 0:Layer_Pixel[layer_i] ] )
        #f.create_dataset('Predicted_NeutralEnergy_Layer' + str(layer_i+1), data=pred_Nu_Layer[ :, layer_i:layer_i+1,  0:Layer_Pixel[layer_i], 0:Layer_Pixel[layer_i] ] )
        f.create_dataset('MidPoint_X_Layer' + str(layer_i+1), data=layer_cell_x[layer_i] )
        f.create_dataset('MidPoint_Y_Layer' + str(layer_i+1), data=layer_cell_x[layer_i] ) 
    #    f.create_dataset('MidPoint_Z_Layer' + str(layer_i+1), data=np.array(zpos_Arr[layer_i]) )

    f.create_dataset('MidPoint_Z_Layer', data=zpos_Arr)
    f.create_dataset('Trk_X_pos', data=Trk_X_pos)
    f.create_dataset('Trk_Y_pos', data=Trk_Y_pos)

    f.create_dataset('Trk_Theta', data=Trk_Theta)
    f.create_dataset('Trk_Phi', data=Trk_Phi)
    f.create_dataset('Smeared_Track_Energy', data=Track_Energy)
    f.create_dataset('Track_Image', data=Track_Layer)
    #f.create_dataset('ChargePi_CellIndex_Layer', data=ChargePi_CellIndex_Layer)
    #f.create_dataset('NeutralPi_CellIndex_Layer', data=NeutralPi_CellIndex_Layer)


    f.create_dataset('Pi0_Theta', data=Pi0_Theta)
    f.create_dataset('Pi0_Phi', data=Pi0_Phi)

    f.create_dataset('Photon1_E', data=Photon1_E)
    f.create_dataset('Photon1_Theta', data=Photon1_Theta)
    f.create_dataset('Photon1_Phi', data=Photon1_Phi)

    f.create_dataset('Photon2_E', data=Photon2_E)
    f.create_dataset('Photon2_Theta', data=Photon2_Theta)
    f.create_dataset('Photon2_Phi', data=Photon2_Phi)

print('Exiting ... Bye!')
# --- end of file ---- #



