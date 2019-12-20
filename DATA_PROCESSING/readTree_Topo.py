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

f = h5py.File('../Outfile_CellInformation_HomDet_15to20GeV_FineGran.h5','r')

#print( f.keys()  )


print ("Init")

#choose kind of topoclusters ("Charged, Total, Neutral")

KIND="Total"

#choose if add noise
AddNoise=False

noise=[13.24,  8.48, 16.95, 13.55,  8.12, 13.67]
layers=[64,64,32,16,16,8]



fnew_ch={}
if AddNoise==False:
    print("Creating calo for cluster on Total Energy + Noise")
    for LAY in range(1,7):
        Calo_T=np.zeros((len(f['Smeared_Track_Energy']), layers[LAY-1], layers[LAY-1]))
        for ev in range(len(f['Smeared_Track_Energy'])):
            Calo_T[ev]=(f['RealRes_'+str(KIND)+'Energy_Layer'+str(LAY)][ev][0])
        fnew_ch['layer_'+str(LAY)]=Calo_T

#out_image = [out_image1,out_image2,out_image3,out_image4,out_image5,out_image6]
#out_image = np.stack(out_image,axis = 0)



if (KIND=="Charged" or KIND =="Neutral") and AddNoise==True:
    for lay in range(1,7):
        Calo_Ch=np.zeros((len(f['Smeared_Track_Energy']), layers[lay-1], layers[lay-1]))
        Calo_Ne=np.zeros((len(f['Smeared_Track_Energy']), layers[lay-1], layers[lay-1]))
        
        for ev in range(len(f['Smeared_Track_Energy'])):
            Calo_Ch[ev] = f['RealRes_'+str(KIND)+'Energy_Layer'+str(lay)][ev][0] + np.random.normal(0, noise[lay-1], (layers[lay-1],layers[lay-1]))
            
        fnew_ch["layer_"+str(lay)]= Calo_Ch
            
out_image = []
i = -1
print ("Looping")
for event in range(len(f['Smeared_Track_Energy'])):

    i = i+ 1
    if i%50==0: print('in loop ev: ',i)
    if i >10: break
    #print(Layer['n1'][i])
    Proto = tools_generic.Clustering( fnew_ch["layer_1"][i],
                                      fnew_ch["layer_2"][i],
                                      fnew_ch["layer_3"][i],
                                      fnew_ch["layer_4"][i],
                                      fnew_ch["layer_5"][i],
                                      fnew_ch["layer_6"][i])


    #print("Final proto",Proto,len(Proto))
    #print('')
    out_image.append(tools_generic.Assign_Topo(Proto))

    



out_image = np.array(out_image)

print(out_image.shape)

print('opening File')
with h5py.File('Outfile_TotalTopo_2N.h5', 'w') as f1:
     for layer_i in range(6):
        f1.create_dataset('TopoClusters' + str(layer_i+1), data=out_image[:, layer_i:layer_i+1,:, :])



if (KIND=="Charged" or KIND =="Neutral") and AddNoise==True:
    print('opening File 2')
    my_h5=h5py.File('Outfile_CellInformation_Noisy'+str(KIND)+'.h5', 'w')


    for lay in range(6):
        my_h5.create_dataset(str(KIND)+'_'+str(lay+1) , data=fnew_ch["layer_"+str(lay+1)])

print ("Exiting ... Bye!")
#
