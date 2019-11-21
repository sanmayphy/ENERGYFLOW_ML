import numpy as np 
import math

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