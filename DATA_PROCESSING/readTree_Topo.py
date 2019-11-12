import ROOT
from sys import exit
import random
from array import array
import numpy as np
#import EventDisplay
import h5py
import tools_generic
#import numpy as np
#from root_numpy import tree2array,array2tree, array2root
ROOT.gROOT.SetBatch(True) 


f = h5py.File('/afs/cern.ch/work/s/sanmay/public/Outfile_CellInformation.h5')

print( f.keys()  )


print ("Init")

RealRes_TotalEnergy_Layer1  = f['RealRes_TotalEnergy_Layer1'][:]
RealRes_TotalEnergy_Layer2  = f['RealRes_TotalEnergy_Layer2'][:]
RealRes_TotalEnergy_Layer3  = f['RealRes_TotalEnergy_Layer3'][:]
RealRes_TotalEnergy_Layer4  = f['RealRes_TotalEnergy_Layer4'][:]
RealRes_TotalEnergy_Layer5  = f['RealRes_TotalEnergy_Layer5'][:]
RealRes_TotalEnergy_Layer6  = f['RealRes_TotalEnergy_Layer6'][:]


#out_image = [out_image1,out_image2,out_image3,out_image4,out_image5,out_image6]
#out_image = np.stack(out_image,axis = 0)
out_image = []
i = 0
print ("Looping")
for event in range(len(RealRes_TotalEnergy_Layer1)):

    i = i+ 1
    if i > 2: break
    Proto = tools_generic.Clustering(RealRes_TotalEnergy_Layer1[i],RealRes_TotalEnergy_Layer2[i],RealRes_TotalEnergy_Layer3[i],RealRes_TotalEnergy_Layer4[i],RealRes_TotalEnergy_Layer5[i],RealRes_TotalEnergy_Layer6[i])
#    print(Proto)
    out_image.append(tools_generic.Assign_Topo(Proto))




out_image = np.array(out_image)
print(out_image.shape)
print('opening File')
with h5py.File('Outfile_TopoCluster.h5', 'w') as f1:
     for layer_i in range(6) :
        f1.create_dataset('TopoClusters' + str(layer_i+1), data=out_image[:, layer_i:layer_i+1,:, :])

print ("Exiting ... Bye!")
#
