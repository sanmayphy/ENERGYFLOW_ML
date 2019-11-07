import os
import sys
import random

import numpy as np
import pandas as pd


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
def SumPixel(image_l, size=64) : 
    
    orig_pixel = image_l.shape[0]
    
    scale = int(orig_pixel/size)
    #print('Scale = ', scale)
    out_image = np.zeros( [orig_pixel, orig_pixel] )
    
    for it_x in range( size ) : 
        for it_y in range( size ) : 
            out_image[it_x:it_x+1, it_y:it_y+1] = np.sum( image_l[scale*it_x:scale*(it_x+1), scale*it_y:scale*(it_y+1)] )
        
    #print('Orig E: ', np.sum(image_l), ', rescaled E: ', np.sum(out_image))   
    return out_image

def MakeRealResolution(image, tr_image) : 
    
    x1 = SumPixel(image[0], size=32)
    #x1 = np.reshape( x1, tuple( [1] + list(x1.shape) ) )
    
    x2 = SumPixel(image[1], size=64)
    #x2 = np.reshape( x2, tuple( [1] + list(x2.shape) ) )
    
    x3 = SumPixel(image[2], size=32)
    #x3 = np.reshape( x3, tuple( [1] + list(x3.shape) ) )
     
    x4 = SumPixel(image[3], size=16)
    #x4 = np.reshape( x4, tuple( [1] + list(x4.shape) ) )
    
    x5 = SumPixel(image[4], size=16)
    #x5 = np.reshape( x5, tuple( [1] + list(x5.shape) ) )
    
    x6 = SumPixel(image[5], size=8)
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

