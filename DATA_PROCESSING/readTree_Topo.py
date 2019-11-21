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

if os.path.exists('Outfile_TopoCluster.h5'):
    os.remove('Outfile_TopoCluster.h5')
else:
    print("Can not delete the file as it doesn't exists")
    print("")
    print('')
f = h5py.File('../Outfile_CellInformation.h5')

print( f.keys()  )


print ("Init")

#Charged Dict
Charged_Layer={}
for LAY in range(1,7):
    Charged_Layer['n'+str(LAY)]=(f['RealRes_ChargedEnergy_Layer'+str(LAY)][:])

#neutral Dict
Neutral_Layer={}
for LAY in range(1,7):
    Neutral_Layer['n'+str(LAY)]=(f['RealRes_NeutralEnergy_Layer'+str(LAY)][:])
#Total dict
Layer={}
for LAY in range(1,7):
    Layer['n'+str(LAY)]=(Charged_Layer['n'+str(LAY)]+Neutral_Layer['n'+str(LAY)])





    
    


#print(RealRes_NeutralEnergy_Layer1, 'layer1', 'AA')
#print(RealRes_NeutralEnergy_Layer1,'ai')
#out_image = [out_image1,out_image2,out_image3,out_image4,out_image5,out_image6]
#out_image = np.stack(out_image,axis = 0)
out_image = []
i = -1
print ("Looping")
for event in range(len(Layer['n1'])):

    i = i+ 1
    if i > 10000: break
    #print(RealRes_NeutralEnergy_Layer1[i], i)
    Proto = tools_generic.Clustering(Layer['n1'][i],Layer['n2'][i],Layer['n3'][i],Layer['n4'][i],Layer['n5'][i],Layer['n6'][i])
    #print('event'+str(event))
    #print("AUA",Proto,len(Proto))
    #print('')
    out_image.append(tools_generic.Assign_Topo(Proto))




out_image = np.array(out_image)
print(out_image.shape)
print('opening File')
with h5py.File('Outfile_TopoCluster.h5', 'w') as f1:
     for layer_i in range(6) :
        f1.create_dataset('TopoClusters' + str(layer_i+1), data=out_image[:, layer_i:layer_i+1,:, :])

print ("Exiting ... Bye!")
#
