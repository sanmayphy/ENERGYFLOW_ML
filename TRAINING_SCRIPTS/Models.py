import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch import tensor
from ResnetUnet import UNetWithResnet50Encoder


true_pix, orig_pix, det_size = 128, 64, 1250.

LayerPix = np.array([64, 32, 32, 16, 16, 8])


def ExpandLayer(layer) : 
    
    size = orig_pix
    out_image = torch.zeros( layer.shape[0], 1, size, size )
    orig_pixel = layer.shape[2]
    
    out_image.permute(2, 3, 1, 0)[0:orig_pixel, 0:orig_pixel, :, :] = layer.permute(2, 3, 1, 0)[0:orig_pixel, 0:orig_pixel, :, :]
    
    return out_image


class Upconv(nn.Module):
    def __init__(self, upscale_factor):
        super(Upconv, self).__init__()

        self.prelu = nn.LeakyReLU(negative_slope=0.1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(256)
        
        self.conv1 = nn.Conv2d(1, 16,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(16, 32,kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(32, 64,kernel_size=3,stride=1,padding=1) 
        self.conv4 = nn.Conv2d(64, 128,kernel_size=3,stride=1,padding=1) 
        self.conv5 = nn.Conv2d(128, 128,kernel_size=3,stride=1,padding=1)
        self.conv6 = nn.Conv2d(128, 256,kernel_size=3,stride=1,padding=1)
        self.conv7 = nn.ConvTranspose2d(256, 1,kernel_size=4,stride=2,padding=1,bias=False)
        self.conv8 = nn.PixelShuffle(upscale_factor)
        self.convf = nn.Conv2d( int(256/upscale_factor**2), 1,kernel_size=1)
    def forward(self, x):
        x = self.bn1(self.prelu(self.conv1(x)))
        x = self.bn2(self.prelu(self.conv2(x)))
        x = self.bn3(self.prelu(self.conv3(x)))
        x = self.bn4(self.prelu(self.conv4(x)))
        x = self.bn5(self.prelu(self.conv5(x)))
        x = self.bn6(self.prelu(self.conv6(x)))
        #print('Shape pre upsampling : ', x.shape)
        x = self.conv8(x)
        x = self.convf(x)
        x = self.relu(x)
        #x.dtype = torch.float64
        #print('Shape post upsampling : ', x.shape)
    
        return x


class SuperResModel(nn.Module):
    
    def __init__(self):
        super(SuperResModel, self).__init__()
        self.cnn1 = Upconv(upscale_factor = int(orig_pix/LayerPix[0]) ).float()
        self.cnn2 = Upconv(upscale_factor = int(orig_pix/LayerPix[1]) ).float()
        self.cnn3 = Upconv(upscale_factor = int(orig_pix/LayerPix[2]) ).float()
        self.cnn4 = Upconv(upscale_factor = int(orig_pix/LayerPix[3]) ).float()
        self.cnn5 = Upconv(upscale_factor = int(orig_pix/LayerPix[4]) ).float()
        self.cnn6 = Upconv(upscale_factor = int(orig_pix/LayerPix[5]) ).float()
        
        self.cnn_comb = nn.Conv2d(7, 6,kernel_size=1,stride=1,padding=0)
        self.unet = UNetWithResnet50Encoder(n_classes=6)#New 21.06
        #self.act_out = nn.ReLU()
#         self.mask_out = nn.ReLU()
        
        self.cnn1_out = nn.Conv2d(1, 1,kernel_size=int(orig_pix/LayerPix[0]),stride=int(orig_pix/LayerPix[0]),padding=0)
        self.cnn2_out = nn.Conv2d(1, 1,kernel_size=int(orig_pix/LayerPix[1]),stride=int(orig_pix/LayerPix[1]),padding=0)
        self.cnn3_out = nn.Conv2d(1, 1,kernel_size=int(orig_pix/LayerPix[2]),stride=int(orig_pix/LayerPix[2]),padding=0)
        self.cnn4_out = nn.Conv2d(1, 1,kernel_size=int(orig_pix/LayerPix[3]),stride=int(orig_pix/LayerPix[3]),padding=0)
        self.cnn5_out = nn.Conv2d(1, 1,kernel_size=int(orig_pix/LayerPix[4]),stride=int(orig_pix/LayerPix[4]),padding=0)
        self.cnn6_out = nn.Conv2d(1, 1,kernel_size=int(orig_pix/LayerPix[5]),stride=int(orig_pix/LayerPix[5]),padding=0)


    def forward(self, x):
        

        out1 = self.cnn1( x[:,0:1, 0:LayerPix[0], 0:LayerPix[0]].float() )

        out2 = self.cnn2( x[:,1:2, 0:LayerPix[1], 0:LayerPix[1]].float() )

        out3 = self.cnn3( x[:,2:3, 0:LayerPix[2], 0:LayerPix[2]].float() )

        out4 = self.cnn4( x[:,3:4, 0:LayerPix[3], 0:LayerPix[3]].float() )

        out5 = self.cnn5( x[:,4:5, 0:LayerPix[4], 0:LayerPix[4]].float() )

        out6 = self.cnn6( x[:,5:6, 0:LayerPix[5], 0:LayerPix[5]].float() )
        
        out7 = x[:, 6:7, 0:orig_pix, 0:orig_pix].float()
        
        # merged_cal  = torch.cat((out1, out2, out3, out4, out5, out6), dim=1)#New 22.06
        merged_im = torch.cat(  (out1, out2, out3, out4, out5, out6, out7), dim=1 )
        
        out_int = self.cnn_comb(merged_im)
        
        out_int = self.unet(out_int)

        
        out1_f = ExpandLayer(  self.cnn1_out(out_int[:, 0:1, :, :])  )
        out2_f = ExpandLayer(  self.cnn2_out(out_int[:, 1:2, :, :])  )
        out3_f = ExpandLayer(  self.cnn3_out(out_int[:, 2:3, :, :])  )
        out4_f = ExpandLayer(  self.cnn4_out(out_int[:, 3:4, :, :])  )
        out5_f = ExpandLayer(  self.cnn5_out(out_int[:, 4:5, :, :])  )
        out6_f = ExpandLayer(  self.cnn6_out(out_int[:, 5:6, :, :])  )
        

        merged =  torch.cat(  (out1_f, out2_f, out3_f, out4_f, out5_f, out6_f), dim=1 )
        

        #merged = torch.clamp(merged, min=-1., max=1.)
        return merged

