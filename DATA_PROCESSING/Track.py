import math
import numpy as np


layer_struct = {'layer1' : 64, 'layer2' : 32, 'layer3' : 32, 'layer4' : 16, 'layer5' : 16, 'layer6' : 8}
def FindCellIndex(tr_x, tr_y, layer_name) : 
    
    add_x, add_y = 0,0
    if( int(tr_x/(1250/128)+ 64) % int( 128/ layer_struct[layer_name]) != 0) : add_x += 1
    if( int(tr_y/(1250/128)+ 64) % int( 128/ layer_struct[layer_name]) != 0) : add_y += 1
    
    cell_x_1 = np.modf( (tr_x/(1250/128)+ 64) * layer_struct[layer_name]/128 )[1].astype(int) + add_x  
    cell_y_1 = np.modf( (tr_y/(1250/128)+ 64) * layer_struct[layer_name]/128 )[1].astype(int) + add_y
    return np.array([cell_x_1, cell_y_1])

def MakeTruthTrajectory(trk_Theta,trk_Phi,trk_X_pos,trk_Y_pos):

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
    z_orig = -1 * (Total_Calo_Length/2 + 150)
    trk_X_n = trk_X_pos/10.0 - np.abs(-zpos_ECAL1+z_orig+3/2*X0_ECAL)*np.tan(trk_Theta)*np.cos(trk_Phi)
    trk_Y_n = trk_Y_pos/10.0 - np.abs(-zpos_ECAL1+z_orig+3/2*X0_ECAL)*np.tan(trk_Theta)*np.sin(trk_Phi)
    
    
    x_1 = (trk_X_n + np.abs(-zpos_ECAL1+z_orig)*np.tan(trk_Theta)*np.cos(trk_Phi))*10
    y_1 = (trk_Y_n + np.abs(-zpos_ECAL1+z_orig)*np.tan(trk_Theta)*np.sin(trk_Phi))*10
    x_2 = (trk_X_n + np.abs(-zpos_ECAL2+z_orig)*np.tan(trk_Theta)*np.cos(trk_Phi))*10
    y_2 = (trk_Y_n + np.abs(-zpos_ECAL2+z_orig)*np.tan(trk_Theta)*np.sin(trk_Phi))*10
    x_3 = (trk_X_n + np.abs(-zpos_ECAL3+z_orig)*np.tan(trk_Theta)*np.cos(trk_Phi))*10
    y_3 = (trk_Y_n + np.abs(-zpos_ECAL3+z_orig)*np.tan(trk_Theta)*np.sin(trk_Phi))*10
    x_4 = (trk_X_n + np.abs(-zpos_HCAL1+z_orig)*np.tan(trk_Theta)*np.cos(trk_Phi))*10
    y_4 = (trk_Y_n + np.abs(-zpos_HCAL1+z_orig)*np.tan(trk_Theta)*np.sin(trk_Phi))*10
    x_5 = (trk_X_n + np.abs(-zpos_HCAL2+z_orig)*np.tan(trk_Theta)*np.cos(trk_Phi))*10
    y_5 = (trk_Y_n + np.abs(-zpos_HCAL2+z_orig)*np.tan(trk_Theta)*np.sin(trk_Phi))*10
    x_6 = (trk_X_n + np.abs(-zpos_HCAL3+z_orig)*np.tan(trk_Theta)*np.cos(trk_Phi))*10
    y_6 = (trk_Y_n + np.abs(-zpos_HCAL3+z_orig)*np.tan(trk_Theta)*np.sin(trk_Phi))*10
    
    l1_idx = FindCellIndex(x_1, y_1, 'layer1')
    l2_idx = FindCellIndex(x_2, y_2, 'layer2')
    l3_idx = FindCellIndex(x_3, y_3, 'layer3')
    l4_idx = FindCellIndex(x_4, y_4, 'layer4')
    l5_idx = FindCellIndex(x_5, y_5, 'layer5')
    l6_idx = FindCellIndex(x_6, y_6, 'layer6')
    
    return np.array([l1_idx, l2_idx, l3_idx, l4_idx, l5_idx, l6_idx]).astype(int)




def Assign_Epred(TotalPred):

    out_image1 = np.zeros( [64,64] )
    out_image2 = np.zeros( [64,64] )
    out_image3 = np.zeros( [64,64] )
    out_image4 = np.zeros( [64,64] )
    out_image5 = np.zeros( [64,64] )
    out_image6 = np.zeros( [64,64] )
    out_image1TC = np.zeros( [64,64] )
    out_image2TC = np.zeros( [64,64] )
    out_image3TC = np.zeros( [64,64] )
    out_image4TC = np.zeros( [64,64] )
    out_image5TC = np.zeros( [64,64] )
    out_image6TC = np.zeros( [64,64] )


    for c in TotalPred:
     if c[0] == 0:
      out_image1[c[1],c[2]] = c[3] 
      out_image1TC[c[1],c[2]] = c[4] 
     if c[0] == 1:
      out_image2[c[1],c[2]] = c[3] 
      out_image2TC[c[1],c[2]] = c[4] 
     if c[0] == 2:
      out_image3[c[1],c[2]] = c[3] 
      out_image3TC[c[1],c[2]] = c[4] 
     if c[0] == 3:
      out_image4[c[1],c[2]] = c[3] 
      out_image4TC[c[1],c[2]] = c[4] 
     if c[0] == 4:
      out_image5[c[1],c[2]] = c[3] 
      out_image5TC[c[1],c[2]] = c[4] 
     if c[0] == 5:
      out_image6[c[1],c[2]] = c[3]  
      out_image6TC[c[1],c[2]] = c[4] 
#     print  out_image2[c[1],c[2]], out_image2TC[c[1],c[2]] 


    out_imagef = [out_image1,out_image2,out_image3,out_image4,out_image5,out_image6]
    out_imagefTC = [out_image1TC,out_image2TC,out_image3TC,out_image4TC,out_image5TC,out_image6TC]
    return out_imagef,out_imagefTC

def Assign_TC(TotalPred,TC,bo):

    out_image1 = np.zeros( [64,64] )
    out_image2 = np.zeros( [64,64] )
    out_image3 = np.zeros( [64,64] )
    out_image4 = np.zeros( [64,64] )
    out_image5 = np.zeros( [64,64] )
    out_image6 = np.zeros( [64,64] )

    for c in TotalPred:
     if bo:
      if c[4] != TC: continue
     if c[0] == 0:
      out_image1[c[1],c[2]] =  TC 
     if c[0] == 1:
      out_image2[c[1],c[2]] =  TC
     if c[0] == 2:
      out_image3[c[1],c[2]] =  TC
     if c[0] == 3:
      out_image4[c[1],c[2]] =  TC
     if c[0] == 4:
      out_image5[c[1],c[2]] =  TC
     if c[0] == 5:
      out_image6[c[1],c[2]] =  TC


    out_imagef = [out_image1,out_image2,out_image3,out_image4,out_image5,out_image6]
    return out_imagef



def ComputeLFI(Topo,track):
  
  layers=[64,32,32,16,16,8]
  En_Lay = np.zeros(6)
  Vol_Cell=Vol_Cells() 


  X0_ECAL = 3.897
  X0_HCAL = 2.357 

  Lambda_int_ECAL = 37.578
  Lambda_int_HCAL = 17.438

  #depth in interaction lengths
  depth=[0, 3*X0_ECAL/Lambda_int_ECAL,16*X0_ECAL/Lambda_int_ECAL,6*X0_ECAL/Lambda_int_ECAL, 1.5, 4.1,1.8 ]


  for cell in Topo:
   #if cell[4] != 0:continue
   X = cell[1]
   Y = cell[2]
   lay = cell[0]
   ga_we = math.exp(-((track[lay][0]-X)**2+(track[lay][1]-Y)**2)/(0.00236672*layers[lay]**2))
   En_Lay[lay]+=(cell[3])*ga_we

  Dens_En = []
  Dens_En.append(0)
  Delta_Dens = []
  for ly in range(6):
     Dens_En.append(En_Lay[ly]/Vol_Cell[ly])
  for i in range(1,7):
     Delta_Dens.append((Dens_En[i]-Dens_En[i-1])/(depth[i]-depth[i-1]))  


  ind=Delta_Dens.index(max(Delta_Dens))
  if Dens_En[ind+1]<Dens_En[ind]: ind-=1
  LFI = ind

  return LFI






def Vol_Cells():
    
    TrLen=125.
    
    X0_CAL=[3.897,3.897,3.897,2.357,2.357,2.357]
    Lambda_int=[37.578,37.578,37.578,17.438,17.438,17.438]
    layers=[64,32,32,16,16,8]
    Depth=[3 *3.897,16 *3.897,6 *3.897,1.5 *17.438,4.1 *17.438,1.8 *17.438]
    #print("Depth in cm: ",Depth)
    
    X0_tran=[TrLen/X0_CAL[i]/layers[i] for i in range(6)]
    
    X0_depth=[Depth[i]/X0_CAL[i] for i in range(6)]
    #print("Pixel size in X0: ", X0_tran)

        
    V_X0=[]
    for i in range(6):
        V_X0.append(X0_tran[i]*X0_tran[i]*X0_depth[i])
        
        
    return V_X0
