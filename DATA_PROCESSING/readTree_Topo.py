#import ROOT
from sys import exit
import random
from array import array
import numpy as np
#import EventDisplay
import h5py
import tools_generic
#import numpy as np
#from root_numpy import tree2array,array2tree, array2root
#ROOT.gROOT.SetBatch(True) 
import os

#if os.path.exists('Outfile_TopoClusterTotal.h5'):
#    os.remove('Outfile_TopoClusterTotal.h5')
#else:
#    print("Can not delete the file as it doesn't exists")
#    print("")
#    print('')
f = h5py.File('../Outfile_CellInformation_HomDet_2to5GeV.h5','r')

#print( f.keys()  )


print ("Init")

#neutral Dict
Layer={}
for LAY in range(1,7):
    Layer['n'+str(LAY)]=(f['RealRes_ChargedEnergy_Layer'+str(LAY)][:])

#print(RealRes_NeutralEnergy_Layer1, 'layer1', 'AA')
#print(RealRes_NeutralEnergy_Layer1,'ai')
#out_image = [out_image1,out_image2,out_image3,out_image4,out_image5,out_image6]
#out_image = np.stack(out_image,axis = 0)
out_image = []
i = -1
print ("Looping")
for event in range(len(Layer['n5'])):

    i = i+ 1
    if i%10==0: print('in loop ev: ',i)
    if i > 1000: break
    #print(Layer['n5'][i])
    Proto = tools_generic.Clustering(Layer['n1'][i],Layer['n2'][i],Layer['n3'][i],Layer['n4'][i],Layer['n5'][i],Layer['n6'][i])
    #print('event'+str(event))

    #print("AUA",Proto,len(Proto))
    #print('')
    out_image.append(tools_generic.Assign_Topo(Proto))




out_image = np.array(out_image)
print(out_image.shape)
print('opening File')
with h5py.File('Outfile_TopoClusterTotal.h5', 'w') as f1:
     for layer_i in range(6):
        f1.create_dataset('TopoClusters' + str(layer_i+1), data=out_image[:, layer_i:layer_i+1,:, :])

print ("Exiting ... Bye!")
#
