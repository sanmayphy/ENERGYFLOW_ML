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


f = uproot.open('/storage/agrp/sanmay/PFLOW_SIMULATION/PFlowNtupleFile_HOMDet_' + sys.argv[1] + 'GeV_Overlap_WS.root')
nameFile = 'Outputfile_V1_Samples' + str(sys.argv[1])



print(f['EventTree'].keys())

# NTotal = len(f['EventTree']['Total_Nu_Energy'][:])

# print('Total Events : ', NTotal)

true_pix, orig_pix, det_size = 128, 64, 1250.



LayerPix = np.array([64, 32, 32, 16, 16, 8])


def SumPixelRed(image_l,  size=64) : 
    
    orig_pixel = image_l.shape[0]
    
    scale = int(orig_pixel/size)
    #print('Scale = ', scale)
    out_image = np.zeros( [size, size] )
    
    for it_x in range( size ) : 
        for it_y in range( size ) : 
            
            val = np.sum( image_l[scale*it_x:scale*(it_x+1), scale*it_y:scale*(it_y+1)] )
                                                  
            
            if(val < 0.) : val = 0
            out_image[it_x:it_x+1, it_y:it_y+1] = val
        
    #print('Orig E: ', np.sum(image_l), ', rescaled E: ', np.sum(out_image))  

    return out_image



def SumPixel(image_l, noise_l, size=64) : 
    
    orig_pixel = image_l.shape[0]
    
    scale = int(orig_pixel/size)
    #print('Scale = ', scale)
    out_image = np.zeros( [orig_pixel, orig_pixel] )
    
    for it_x in range( size ) : 
        for it_y in range( size ) : 
            
            val = np.sum( image_l[scale*it_x:scale*(it_x+1), scale*it_y:scale*(it_y+1)] )                                                  + noise_l[it_x:it_x+1, it_y:it_y+1]
            
            if(val < 0.) : val = 0
            out_image[it_x:it_x+1, it_y:it_y+1] = val
        
    #print('Orig E: ', np.sum(image_l), ', rescaled E: ', np.sum(out_image))  

    return out_image



def MakeReducedResolution(image) : 
    
    x1 = SumPixelRed(image[0], size= orig_pix )
    #print('X1 shape : ', x1.shape)
    
    x2 = SumPixelRed(image[1], size= orig_pix )
    #print('X2 shape : ', x2.shape)
    
    x3 = SumPixelRed(image[2], size= orig_pix )
    #print('X3 shape : ', x3.shape)
     
    x4 = SumPixelRed(image[3], size= orig_pix )
    #print('X4 shape : ', x4.shape)
    
    x5 = SumPixelRed(image[4], size= orig_pix )
    #print('X5 shape : ', x5.shape)
    
    x6 = SumPixelRed(image[5], size= orig_pix )
    #print('X6 shape : ', x6.shape)
    
    x = [ x1, x2, x3, x4, x5, x6]
    return np.stack(x, axis=0)


import scipy
from scipy.ndimage import gaussian_filter1d

def SmearImage(image) : 
    
    out_image = np.zeros(image.shape)
    NLayer = image.shape[0]
    
    NPixel = image.shape[1]
    
    for layer_i in range(NLayer) :
        for xbin_j in range(NPixel) : 
            for ybin_k in range(NPixel) : 
                if(image[layer_i][xbin_j][ybin_k] > 0.) : 
                   # out_image[layer_i][xbin_j][ybin_k] = gaussian_filter1d( [image[layer_i][xbin_j][ybin_k] ], np.sqrt(image[layer_i][xbin_j][ybin_k]) * 4 )
                    out_image[layer_i][xbin_j][ybin_k] = np.maximum( 0., np.random.normal(image[layer_i][xbin_j][ybin_k], np.sqrt(image[layer_i][xbin_j][ybin_k]) * 4, 1)[0] )
                    #x = np.random.normal(image[layer_i][xbin_j][ybin_k], np.sqrt(image[layer_i][xbin_j][ybin_k]) * 4, 1)
                    #print (x[0])
                    
    return out_image


def MakeTrackLayer(tr_e, tr_x, tr_y) : 
    
    out_image = np.zeros( [orig_pix, orig_pix] )
    
    x_idx = int(  (tr_x + det_size/2.)/det_size * orig_pix  )
    y_idx = int(  (tr_y + det_size/2.)/det_size * orig_pix  )
    
    out_image[x_idx, y_idx] = tr_e
    
    #out_image = out_image.reshape([1, 64, 64])
    
    return out_image


def MakeRealResolution(image, noise_image, tr_image, doTopo=False) : 
    
    x1 = SumPixel(image[0], noise_image[0], size= LayerPix[0] )
    #x1 = np.reshape( x1, tuple( [1] + list(x1.shape) ) )
    
    x2 = SumPixel(image[1], noise_image[1], size= LayerPix[1] )
    #x2 = np.reshape( x2, tuple( [1] + list(x2.shape) ) )
    
    x3 = SumPixel(image[2], noise_image[2], size= LayerPix[2] )
    #x3 = np.reshape( x3, tuple( [1] + list(x3.shape) ) )
     
    x4 = SumPixel(image[3], noise_image[3], size= LayerPix[3] )
    #x4 = np.reshape( x4, tuple( [1] + list(x4.shape) ) )
    
    x5 = SumPixel(image[4], noise_image[4], size= LayerPix[4] )
    #x5 = np.reshape( x5, tuple( [1] + list(x5.shape) ) )
    
    x6 = SumPixel(image[5], noise_image[5], size= LayerPix[5] )
    #x6 = np.reshape( x6, tuple( [1] + list(x6.shape) ) )


    # ----- make the topocluster ------- #
    topo_im = np.zeros([6, orig_pix, orig_pix])

    if(doTopo) : 
        Proto = tools_generic.Clustering( x1[ 0:LayerPix[0], 0:LayerPix[0]  ], 
                                          x2[ 0:LayerPix[1], 0:LayerPix[1]  ],
                                          x3[ 0:LayerPix[2], 0:LayerPix[2]  ],
                                          x4[ 0:LayerPix[3], 0:LayerPix[3]  ],
                                          x5[ 0:LayerPix[4], 0:LayerPix[4]  ],
                                          x6[ 0:LayerPix[5], 0:LayerPix[5]  ],
                                        )
      
        topo_clus = tools_generic.Assign_Topo(Proto) 

        topo_clus = np.array(topo_clus)

        topo_clus[ np.where(topo_clus > 0.) ] = 1.

        topo_im = topo_clus

        x1 = x1 * topo_im[0]
        x2 = x2 * topo_im[1]
        x3 = x3 * topo_im[2]
        x4 = x4 * topo_im[3]
        x5 = x5 * topo_im[4]
        x6 = x6 * topo_im[5]
    
    x = [ x1, x2, x3, x4, x5, x6 , tr_image ]

    #x.extend( topo_im )

    return np.stack(x, axis=0)



def MakeFraction(nu_image, image) : 
        
    NLayer = image.shape[0] - 1
    
    out_image = np.zeros( [NLayer, image.shape[1], image.shape[2] ] )

    for layer_i in range(NLayer) : 
        #if(np.sum(image[layer_i]) > 0.) :
        #topo_im =  image[layer_i + 7]
        out_image[layer_i] = (nu_image[layer_i])/image[layer_i]
        
        #loc_neg = np.where(image[layer_i] <= 0.)
        #loc_ge =  np.where(image[layer_i] < nu_image[layer_i])
        
            #print( np.argwhere( np.isnan(out_image[layer_i]) ) )
            #out_image[layer_i][ np.argwhere( np.isnan(out_image[layer_i]) ) ] = 0.

    out_image[ np.where(  np.isinf(out_image) ) ]   = 0.            
    out_image[ np.where(  np.isnan(out_image) ) ]   = 0. 
    out_image[ np.where(  out_image <= 0. ) ]   = 0.
    out_image[ np.where(  out_image >= 1. ) ]   = 1.
    
    
    return out_image



# -------- starting the event loops ------ #


NumberOfAllEvents = 6000
NEvent = 6000

NSlice = 1000

NStart = 100000

for j in range(NumberOfAllEvents//NEvent):


    Ch_Layer_Orig = [f['EventTree'].array('cellCh_Energy', entrystart = NStart + j * NEvent+ i*NSlice, entrystop = NStart +j * NEvent + (i+1)*NSlice) for i in range(0,int(NEvent/NSlice))]
    Ch_Layer_Orig = np.stack(Ch_Layer_Orig, axis=0)
    Ch_Layer_Orig = Ch_Layer_Orig.reshape(NEvent, 6, true_pix, true_pix)


    Nu_Layer_Orig = [f['EventTree'].array('cellNu_Energy', entrystart = NStart + j * NEvent+ i*NSlice, entrystop = NStart + j * NEvent + (i+1)*NSlice) for i in range(0,int(NEvent/NSlice))]
    Nu_Layer_Orig = np.stack(Ch_Layer_Orig, axis=0)
    Nu_Layer_Orig = Nu_Layer_Orig.reshape(NEvent, 6, true_pix, true_pix)

    Ch_Layer = np.array([  MakeReducedResolution(i_image) for i_image in Ch_Layer_Orig ])
    Nu_Layer = np.array([  MakeReducedResolution(i_image) for i_image in Nu_Layer_Orig ])

    del Ch_Layer_Orig; del Nu_Layer_Orig;

    #print('Ch_Layer shape : ',  Ch_Layer.shape )


    Smeared_Nu_Layer = np.array([ SmearImage(image) for image in  Nu_Layer ])


    Smeared_Ch_Layer = np.array([ SmearImage(image) for image in  Ch_Layer ])


    # ----- total energy = smeared charged + smeared neutral ----- #
    Layer = Smeared_Ch_Layer + Smeared_Nu_Layer


    del Ch_Layer; del Nu_Layer; del Smeared_Ch_Layer;


    Noise_Layer = [f['EventTree'].array('Noise_cell_Energy', entrystart = NStart + j * NEvent+ i*NSlice, entrystop = NStart + j * NEvent + (i+1)*NSlice) for i in range(0,int(NEvent/NSlice))]
    Noise_Layer = np.stack(Noise_Layer, axis=0)
    Noise_Layer = Noise_Layer.reshape(NEvent, 6, true_pix, true_pix)

    # --- adding the track --- #
    Trk_X_pos = [f['EventTree'].array('Trk_X_pos', entrystart = NStart + j * NEvent+ i*NSlice, entrystop = NStart + j * NEvent + (i+1)*NSlice) for i in range(0, int(NEvent/NSlice)  )]
    Trk_Y_pos = [f['EventTree'].array('Trk_Y_pos', entrystart = NStart + j * NEvent+ i*NSlice, entrystop = NStart + j * NEvent + (i+1)*NSlice) for i in range(0, int(NEvent/NSlice)  )]


    Trk_X_pos = np.stack(Trk_X_pos, axis=0)
    Trk_X_pos = Trk_X_pos.reshape(NEvent, )


    Trk_Y_pos = np.stack(Trk_Y_pos, axis=0)
    Trk_Y_pos = Trk_Y_pos.reshape(NEvent, )

    Track_Energy = [f['EventTree'].array('Smeared_Ch_Energy', entrystart = NStart + j * NEvent+ i*NSlice, entrystop = NStart + j * NEvent + (i+1)*NSlice) for i in range(0,  int(NEvent/NSlice)  )]


    Track_Energy = np.stack(Track_Energy, axis=0)
    Track_Energy = Track_Energy.reshape(NEvent, )

    Track_Layer = np.array( [ MakeTrackLayer(Track_Energy[it], Trk_X_pos[it], Trk_Y_pos[it]) for  it in range(len(Track_Energy))] )

    del Trk_X_pos; del Trk_Y_pos



    train_size, val_size = int(0.0 * NEvent), int(0.0 * NEvent)
    test_size = NEvent - (train_size + val_size)


    indices = list(range(NEvent))
    np.random.shuffle(indices)



    Layer, Smeared_Nu_Layer, Noise_Layer = Layer[indices],Smeared_Nu_Layer[indices], Noise_Layer[indices]


    Track_Layer = Track_Layer[indices]



    train_Layer, val_Layer, test_Layer = Layer[0:train_size], Layer[train_size: train_size + val_size], Layer[train_size + val_size : train_size + val_size + test_size]

    train_Noise_Layer, val_Noise_Layer, test_Noise_Layer = Noise_Layer[0:train_size], Noise_Layer[train_size: train_size + val_size], Noise_Layer[train_size + val_size : train_size + val_size + test_size]


    train_Nu_Layer, val_Nu_Layer, test_Nu_Layer = Smeared_Nu_Layer[0:train_size], Smeared_Nu_Layer[train_size: train_size + val_size], Smeared_Nu_Layer[train_size + val_size : train_size + val_size + test_size]
    train_Track_Layer, val_Track_Layer, test_Track_Layer = Track_Layer[0:train_size], Track_Layer[train_size: train_size + val_size], Track_Layer[train_size + val_size : train_size + val_size + test_size]

    train_RealRes = np.array([ MakeRealResolution(train_Layer[i_img], train_Noise_Layer[i_img], train_Track_Layer[i_img], doTopo=True) for i_img in range( len(train_Layer) )  ])

    test_RealRes = np.array([ MakeRealResolution(test_Layer[i_img], test_Noise_Layer[i_img], test_Track_Layer[i_img], doTopo=True) for i_img in range( len(test_Layer) )  ])
    val_RealRes = np.array([ MakeRealResolution(val_Layer[i_img], val_Noise_Layer[i_img], val_Track_Layer[i_img], doTopo=True) for i_img in range( len(val_Layer) )  ])


    train_Nu_RealRes = np.array([ MakeRealResolution(train_Nu_Layer[i_img], np.zeros([6, orig_pix, orig_pix]), train_Track_Layer[i_img]) for i_img in range( len(train_Nu_Layer) )  ])
    test_Nu_RealRes = np.array([ MakeRealResolution(test_Nu_Layer[i_img], np.zeros([6, orig_pix, orig_pix]), test_Track_Layer[i_img]) for i_img in range( len(test_Nu_Layer) )  ])
    val_Nu_RealRes = np.array([ MakeRealResolution(val_Nu_Layer[i_img], np.zeros([6, orig_pix, orig_pix]), val_Track_Layer[i_img]) for i_img in range( len(val_Nu_Layer) )  ])


    del Layer; del train_Layer; del val_Layer; del test_Layer;
    del Track_Layer; del train_Track_Layer; del val_Track_Layer; del test_Track_Layer;
    del Noise_Layer; del train_Noise_Layer; del val_Noise_Layer; del test_Noise_Layer;
    del train_Nu_Layer; del val_Nu_Layer; del test_Nu_Layer; 

    Pair_train_Nu_Layer = np.array( list(zip(train_Nu_RealRes, train_RealRes) ) )
    Pair_val_Nu_Layer = np.array( list(zip(val_Nu_RealRes, val_RealRes) ) )
    Pair_test_Nu_Layer = np.array( list(zip(test_Nu_RealRes, test_RealRes) ) )


    frac_train_Nu_Layer = np.array([ MakeFraction(image[0], image[1]) for image in Pair_train_Nu_Layer ])
    frac_val_Nu_Layer = np.array([ MakeFraction(image[0], image[1]) for image  in Pair_val_Nu_Layer ])
    frac_test_Nu_Layer = np.array([ MakeFraction(image[0], image[1]) for image  in Pair_test_Nu_Layer ])
    

    # In[ ]:


    print('Frac shape : ', frac_train_Nu_Layer.shape)


    if (j==0):
        hfile = h5py.File('/storage/agrp/sanmay/PFLOW_ANALYSIS/RectangularGeo/CROSS_CHECK/'+nameFile+'.hdf5','w')



        #traingrp = hfile.create_group("train")
        #validgrp = hfile.create_group("valid")
        testgrp = hfile.create_group("test")


        '''
        dsetrainin = traingrp.create_dataset("input" , data = train_RealRes,       compression = "lzf", chunks=(10, 6, 256, 256), maxshape=(140000,7,256,256))
        dsetrainou = traingrp.create_dataset("output", data = frac_train_Nu_Layer, compression = "lzf", chunks=(10, 6, 256, 256), maxshape=(140000,6,256,256))
        # dsetrainou = traingrp.create_dataset("output_noise", data = frac_train_NuNo_Layer, compression = "lzf", chunks=(10, 6, 256, 256), maxshape=(None,6,256,256))
        dsetvaliin = validgrp.create_dataset("input" , data = val_RealRes,         compression = "lzf", chunks=(10, 6, 256, 256), maxshape=(40000,7,256,256))
        dsetvaliou = validgrp.create_dataset("output", data = frac_val_Nu_Layer,   compression = "lzf", chunks=(10, 6, 256, 256), maxshape=(40000,6,256,256))
        '''
        # dsetvaliou = validgrp.create_dataset("output_noise", data = frac_val_NuNo_Layer,   compression = "lzf", chunks=(10, 6, 256, 256), maxshape=(None,6,256,256))
        dsettestin = testgrp.create_dataset ("input" , data = test_RealRes,        compression = "lzf", chunks=(10, 6, 256, 256), maxshape=(20000,7,256,256))
        dsettestou = testgrp.create_dataset ("output", data = frac_test_Nu_Layer,  compression = "lzf", chunks=(10, 6, 256, 256), maxshape=(20000,6,256,256))
        # dsettestou = testgrp.create_dataset ("output_noise", data = frac_test_NuNo_Layer,  compression = "lzf", chunks=(10, 6, 256, 256), maxshape=(None,6,256,256))
        hfile.close()
    else:
        with h5py.File('/storage/agrp/sanmay/PFLOW_ANALYSIS/RectangularGeo/CROSS_CHECK/'+nameFile+'.hdf5', 'a') as hf:
            '''
            hf["train"]["input"].resize((hf["train"]["input"].shape[0] + train_RealRes.shape[0]), axis = 0)
            hf["train"]["input"][-train_RealRes.shape[0]:] = train_RealRes
            hf["train"]["output"].resize((hf["train"]["output"].shape[0] + frac_train_Nu_Layer.shape[0]), axis = 0)
            hf["train"]["output"][-frac_train_Nu_Layer.shape[0]:] = frac_train_Nu_Layer
            # hf["train"]["output_noise"].resize((hf["train"]["output_noise"].shape[0] + frac_train_NuNo_Layer.shape[0]), axis = 0)
            # hf["train"]["output_noise"][-frac_train_NuNo_Layer.shape[0]:] = frac_train_NuNo_Layer

            hf["valid"]["input"].resize((hf["valid"]["input"].shape[0] + val_RealRes.shape[0]), axis = 0)
            hf["valid"]["input"][-val_RealRes.shape[0]:] = val_RealRes
            hf["valid"]["output"].resize((hf["valid"]["output"].shape[0] + frac_val_Nu_Layer.shape[0]), axis = 0)
            hf["valid"]["output"][-frac_val_Nu_Layer.shape[0]:] = frac_val_Nu_Layer
            # hf["valid"]["output_noise"].resize((hf["valid"]["output_noise"].shape[0] + frac_val_NuNo_Layer.shape[0]), axis = 0)
            # hf["valid"]["output_noise"][-frac_val_NuNo_Layer.shape[0]:] = frac_val_NuNo_Layer
            '''
            hf["test"]["input"].resize((hf["test"]["input"].shape[0] + test_RealRes.shape[0]), axis = 0)
            hf["test"]["input"][-test_RealRes.shape[0]:] = test_RealRes
            hf["test"]["output"].resize((hf["test"]["output"].shape[0] + frac_test_Nu_Layer.shape[0]), axis = 0)
            hf["test"]["output"][-frac_test_Nu_Layer.shape[0]:] = frac_test_Nu_Layer
            # hf["test"]["output_noise"].resize((hf["test"]["output_noise"].shape[0] + frac_test_NuNo_Layer.shape[0]), axis = 0)
            # hf["test"]["output_noise"][-frac_test_NuNo_Layer.shape[0]:] = frac_test_NuNo_Layer
    del frac_train_Nu_Layer, frac_val_Nu_Layer, frac_test_Nu_Layer, train_RealRes, val_RealRes, test_RealRes
    print ('Event : ', j*NEvent)

