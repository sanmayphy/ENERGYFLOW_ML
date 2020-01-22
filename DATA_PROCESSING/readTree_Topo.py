#import ROOT
#from sys import exit
import sys
import random
from array import array
import numpy as np
import h5py
import tools_generic
import os


#f = h5py.File('../File_En/Outfile_CellInformation_HomDet_15to20GeV_FineGran128_V1.h5','r')
f = h5py.File('../File_En/Outfile_CellInformation_HomDet_2to5GeV_FineGran128_V1.h5','r')

print ("Init")



KIND="Total"       #raw_input("Set the Topoclustering Kind (Charged, Total, Neutral): ")    #Set kind of clustering ("Charged, Total, Neutral")  



AddNoise=True
if KIND=="Total":
    AddNoise=False              


N_EV=len(f['Smeared_Track_Energy'])
noise=[13, 34, 17, 14,  8, 14]
layers=[64, 32, 32, 16, 16, 8]


fnew_ch={}
if AddNoise==False:
    print("Creating calo for cluster on Total Energy + Noise")
    for LAY in range(1,7):
        Calo_T=np.zeros((N_EV, layers[LAY-1], layers[LAY-1]))
        for ev in range(N_EV):
            Calo_T[ev]=(f['RealRes_'+str(KIND)+'Energy_Layer'+str(LAY)][ev][0])
        fnew_ch['layer_'+str(LAY)]=Calo_T

#out_image = [out_image1,out_image2,out_image3,out_image4,out_image5,out_image6]
#out_image = np.stack(out_image,axis = 0)

if (KIND=="Charged" or KIND =="Neutral") and AddNoise==True:
    print("Creating calo for cluster on "+str(KIND)+" Energy + Noise")
    for lay in range(1,7):
        Calo=np.zeros((N_EV, layers[lay-1], layers[lay-1]))
        
        for ev in range(N_EV):
            Calo[ev] = f['RealRes_'+str(KIND)+'Energy_Layer'+str(lay)][ev][0] + f['RealRes_Noise_Layer'+str(lay)][ev][0]
            
        fnew_ch["layer_"+str(lay)]= Calo

    
out_image = []
i = -1
print ("Looping")
for event in range(N_EV):

    i = i+ 1
    if i%100==0: print('in loop ev: ',i)
    if i >20: break
    #print(Layer['n1'][i])
    Proto = tools_generic.Clustering( fnew_ch["layer_1"][i],
                                      fnew_ch["layer_2"][i],
                                      fnew_ch["layer_3"][i],
                                      fnew_ch["layer_4"][i],
                                      fnew_ch["layer_5"][i],
                                      fnew_ch["layer_6"][i])
    print('---------- Final Topocluster '+str(i)+' -----------')
    print("Final proto",Proto,len(Proto),)
    #print('')
    out_image.append(tools_generic.Assign_Topo(Proto))

out_image = np.array(out_image)

print(out_image.shape)

print('opening File')
#with h5py.File('Outfile_15to20GeV_'+str(KIND)+'Topo.h5', 'w') as f1:
with h5py.File('Outfile_2to5GeV_'+str(KIND)+'Topo.h5', 'w') as f1:

    for layer_i in range(6):
        f1.create_dataset('TopoClusters' + str(layer_i+1), data=out_image[:, layer_i:layer_i+1,:, :])

print ("Exiting ... Bye!")
#
