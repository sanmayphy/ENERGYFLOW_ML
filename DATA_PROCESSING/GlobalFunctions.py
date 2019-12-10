import os
import sys
import random

import numpy as np
import pandas as pd

import math

# ------ hard coding the geometry for truth trajectory ---------- #
X0_ECAL = 3.897
Lambda_int_HCAL = 17.438

Total_ECAL_Length = 3 * X0_ECAL + 16 * X0_ECAL + 6 * X0_ECAL
Total_HCAL_Length = 1.5 * Lambda_int_HCAL + 4.1 * Lambda_int_HCAL + 1.8 * Lambda_int_HCAL
Total_Calo_Length = Total_ECAL_Length + Total_HCAL_Length + 1.0 # -- there is a 1 cm gap between ECAL & HCAL


zpos_ECAL1 = -1 * Total_Calo_Length/2 + 3 * X0_ECAL/2
zpos_ECAL2 = zpos_ECAL1 + 3 * X0_ECAL/2 + 16  * X0_ECAL/2
zpos_ECAL3 = zpos_ECAL2 + 16 * X0_ECAL/2 + 6 * X0_ECAL/2

zpos_GAP = zpos_ECAL3 + 6 * X0_ECAL/2 + 1/2

zpos_HCAL1 = zpos_GAP + 1/2 + 1.5 * Lambda_int_HCAL/2
zpos_HCAL2 = zpos_HCAL1 + 1.5 * Lambda_int_HCAL/2 + 4.1 * Lambda_int_HCAL/2
zpos_HCAL3 = zpos_HCAL2 + 4.1 * Lambda_int_HCAL/2 + 1.8 * Lambda_int_HCAL/2

layer_struct = {'layer1' : 32, 'layer2' : 64, 'layer3' : 32, 'layer4' : 16, 'layer5' : 16, 'layer6' : 8}


# ---- finding the cell indices for extrapolated pi0/pi+ --------------------- #
def FindCellIndex(tr_x, tr_y, layer_name) : 
    
    # out_image = np.zeros( [layer_struct[layer_name], layer_struct[layer_name]] )
    
    x_idx = int(  (tr_x + 125)/250 *  layer_struct[layer_name] )
    y_idx = int(  (tr_y + 125)/250 * layer_struct[layer_name]  )
    
    return np.array([x_idx, y_idx])


def MakeTruthTrajectory(Theta, Phi, x_indx, y_indx) : 

    x_orig, y_orig = 0., 0.
    z_orig = -1 * (Total_Calo_Length/2 + 250)

    if(x_indx == 5 and y_indx == 5) : x_orig, y_orig   =  20.,  20.
    if(x_indx == 5 and y_indx == -5) : x_orig, y_orig  =  20., -20.
    if(x_indx == -5 and y_indx == 5) : x_orig, y_orig  = -20.,  20.
    if(x_indx == -5 and y_indx == -5) : x_orig, y_orig = -20., -20.

    layer1_x = x_orig + (zpos_ECAL1 - z_orig) * math.tan(Theta) * math.cos(Phi)
    layer1_y = y_orig + (zpos_ECAL1 - z_orig) * math.tan(Theta) * math.sin(Phi)

    layer2_x = x_orig + (zpos_ECAL2 - z_orig) * math.tan(Theta) * math.cos(Phi)
    layer2_y = y_orig + (zpos_ECAL2 - z_orig) * math.tan(Theta) * math.sin(Phi)

    layer3_x = x_orig + (zpos_ECAL3 - z_orig) * math.tan(Theta) * math.cos(Phi)
    layer3_y = y_orig + (zpos_ECAL3 - z_orig) * math.tan(Theta) * math.sin(Phi)

    layer4_x = x_orig + (zpos_HCAL1 - z_orig) * math.tan(Theta) * math.cos(Phi)
    layer4_y = y_orig + (zpos_HCAL1 - z_orig) * math.tan(Theta) * math.sin(Phi)

    layer5_x = x_orig + (zpos_HCAL2 - z_orig) * math.tan(Theta) * math.cos(Phi)
    layer5_y = y_orig + (zpos_HCAL2 - z_orig) * math.tan(Theta) * math.sin(Phi)

    layer6_x = x_orig + (zpos_HCAL3 - z_orig) * math.tan(Theta) * math.cos(Phi)
    layer6_y = y_orig + (zpos_HCAL3 - z_orig) * math.tan(Theta) * math.sin(Phi)

    CellIndexLayer = []

    CellIndexLayer.append(  FindCellIndex(layer1_x, layer1_y, 'layer1') )
    CellIndexLayer.append(  FindCellIndex(layer2_x, layer2_y, 'layer2') )
    CellIndexLayer.append(  FindCellIndex(layer3_x, layer3_y, 'layer3') )
    CellIndexLayer.append(  FindCellIndex(layer4_x, layer4_y, 'layer4') )
    CellIndexLayer.append(  FindCellIndex(layer5_x, layer5_y, 'layer5') )
    CellIndexLayer.append(  FindCellIndex(layer6_x, layer6_y, 'layer6') )

    return np.array(CellIndexLayer)


# ------ making a track image of size 64 X 64 from energy and x,y position ----- #
def MakeTrackLayer(tr_e, tr_x, tr_y) : 
    
    out_image = np.zeros( [64, 64] )
    
    x_idx = int(  (tr_x + 1250)/2500 * 64  )
    y_idx = int(  (tr_y + 1250)/2500 * 64  )
    
    out_image[x_idx, y_idx] = tr_e
    
    #out_image = out_image.reshape([1, 64, 64])
    
    return out_image
# ------------------------------------------------------------------------------ #


# ------ making variable resolution from uniform resolution ----- #
def SumPixel(image_l, noise_l, size=64) : 
    
    orig_pixel = image_l.shape[0]
    
    scale = int(orig_pixel/size)
    #print('Scale = ', scale)
    out_image = np.zeros( [orig_pixel, orig_pixel] )
    
    for it_x in range( size ) : 
        for it_y in range( size ) : 
            
            val = np.sum( image_l[scale*it_x:scale*(it_x+1), scale*it_y:scale*(it_y+1)] )\
                                                  + noise_l[it_x:it_x+1, it_y:it_y+1]
            
            if(val < 0.) : val = 0
            out_image[it_x:it_x+1, it_y:it_y+1] = val
        
    #print('Orig E: ', np.sum(image_l), ', rescaled E: ', np.sum(out_image))  

    return out_image

def MakeRealResolution(image, noise_image, tr_image) : 
    
    x1 = SumPixel(image[0], noise_image[0], size=32)
    #x1 = np.reshape( x1, tuple( [1] + list(x1.shape) ) )
    
    x2 = SumPixel(image[1], noise_image[1], size=64)
    #x2 = np.reshape( x2, tuple( [1] + list(x2.shape) ) )
    
    x3 = SumPixel(image[2], noise_image[2], size=32)
    #x3 = np.reshape( x3, tuple( [1] + list(x3.shape) ) )
     
    x4 = SumPixel(image[3], noise_image[3], size=16)
    #x4 = np.reshape( x4, tuple( [1] + list(x4.shape) ) )
    
    x5 = SumPixel(image[4], noise_image[4], size=16)
    #x5 = np.reshape( x5, tuple( [1] + list(x5.shape) ) )
    
    x6 = SumPixel(image[5], noise_image[5], size=8)
    #x6 = np.reshape( x6, tuple( [1] + list(x6.shape) ) )
    
    x = [ x1, x2, x3, x4, x5, x6 , tr_image]
    return np.stack(x, axis=0)
# ------------------------------------------------------------------------------ #

# ---- make energy fraction from total and partial energy distribution --------- #
def MakeFraction(nu_image, image) : 
    
    out_image = np.zeros( [nu_image.shape[0], nu_image.shape[1], nu_image.shape[2] ] )
    NLayer = nu_image.shape[0]
    
    for layer_i in range(NLayer) : 
        if(np.sum(image[layer_i]) > 0.) : 
            out_image[layer_i] = nu_image[layer_i]/image[layer_i]
            
            #print( np.argwhere( np.isnan(out_image[layer_i]) ) )
            #out_image[layer_i][ np.argwhere( np.isnan(out_image[layer_i]) ) ] = 0.
            
    out_image[ np.where(  np.isnan(out_image) ) ]   = 0.     
    return out_image
# ------------------------------------------------------------------------------ #

# -- make partial energy fraction from total energy distribution and fraction -- #
def MakeEnergyImage(nu_fraction, image) : 
    
    out_image = np.zeros( [image.shape[0]-1, image.shape[1], image.shape[2] ] )
    NLayer = image.shape[0] - 1
    
    for layer_i in range(NLayer) : 
        if(np.sum(image[layer_i]) > 0.) : 
            
            loc = np.where( image[layer_i] > 10 )
            out_image[layer_i][loc] = nu_fraction[layer_i][loc] * image[layer_i][loc]
            
            #print( np.argwhere( np.isnan(out_image[layer_i]) ) )
            #out_image[layer_i][ np.argwhere( np.isnan(out_image[layer_i]) ) ] = 0.
            
    out_image[ np.where(  np.isnan(out_image) ) ]   = 0.     
    return out_image


# ------------------------------------------------------------------------------ #

