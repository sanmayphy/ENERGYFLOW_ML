import sys
import random
import ROOT
from array import array
import numpy as np
import h5py
import tools_generic
import os
import math
import Track
from sklearn.cluster import KMeans
from ROOT import TLorentzVector


layers=[64,32,32,16,16,8]
#TODO need the initial shooting information
or_X = 0
or_Y = 0


def pos_cm(pos_pix,gran):
  det_size = 125.  
  return (pos_pix + 1/2 - gran/2) * det_size / gran

def Mass_distance(pos_,ene_):
  Mass = 0
  Distance = 0

  ## check the two populations         
  kmean_ = KMeans(n_clusters=2, init='k-means++', max_iter=1000, n_init=100, random_state=0)
  predi_ = kmean_.fit(pos_,ene_)
  

  ene_test=np.asarray(ene_)
  ph_1 = kmean_.cluster_centers_[0]
  ph_2 = kmean_.cluster_centers_[1]
  #These are the energy of the two populations
  E_ph_1 = np.sum(ene_test[np.where(predi_.labels_==1)])
  E_ph_2 = np.sum(ene_test[np.where(predi_.labels_==0)])
  #Now we need to define the px, py, px momenta
  ######WARNING we do not have information about the original position########
  DX1 = ph_1[0] - or_X
  DY1 = ph_1[1] - or_Y
  DZ1 = ph_1[2]*300.

  DX2 = ph_2[0] - or_X
  DY2 = ph_2[1] - or_Y
  DZ2 = ph_2[2]*300.

  r1 = np.sqrt(DX1**2 + DY1**2 + DZ1**2)
  r2 = np.sqrt(DX2**2 + DY2**2 + DZ2**2)
  
  corrx   = DX1 / r1
  corry   = DY1 / r1
  corrz   = DZ1 / r1
  corrx_2 = DX2 / r2
  corry_2 = DY2 / r2
  corrz_2 = DZ2 / r2

  photon1 = TLorentzVector(E_ph_1*corrx,   E_ph_1*corry,  E_ph_1*corrz ,   E_ph_1)
  photon2 = TLorentzVector(E_ph_2*corrx_2, E_ph_2*corry_2,E_ph_2*corrz_2 , E_ph_2)
  Mass = (photon1+photon2).Mag()
  distance = np.sqrt( (DX1-DX2)**2 + (DY1-DY2)**2 + (DZ1-DZ2)**2 )

#  print("Mass is: ",(photon1+photon2).Mag(),distance)

  return Mass, distance

#############################
# Def useful for Kmean mass #
#############################




X0 = 3.897
LI = 17.438

num_L = 3
depth=np.asarray([3*X0, 16*X0, 6*X0] )
orig = 150.
Z_L=np.asarray([150+np.sum(depth[:l]) for l in range( num_L )])
Z_L[3:]+=1
tr_dim = 125.
print("Position beginning of layers: ", Z_L)
#########################

Events = 1000

fnn = h5py.File('/eos/user/s/sanmay/public/MULTIPLE_TRAINING/PredictionFile_SuperRes.h5')

print(list(fnn.keys()))


ED_n_Plot = []
ED_nt_Plot = []
ED_n_Plot_rescale = []
ED_nt_Plot_rescale = []
#averageDisaplay
h_reco_mass =  ROOT.TH1D("h_reco_mass","h_reco_mass",100,0,1000)
h_truth_mass =  ROOT.TH1D("h_truth_mass","h_truth_mass",100,0,1000)
h_truth_mass_sta =  ROOT.TH1D("h_truth_mass_sta","h_truth_mass_sta",100,0,1000)

h_reco_distance  =  ROOT.TH1D("h_reco_dist","h_reco_dist",  25,0,50)
h_truth_distance =  ROOT.TH1D("h_truth_dist","h_truth_dist",25,0,50)
for lay in range(3):
  print(lay)

  h_nt_ED =  ROOT.TProfile2D("h_nt_ED"+str(lay),"h_nt_ED"+str(lay),64*2,-64,64,64*2,-64,64)
  ED_nt_Plot.append(h_nt_ED)
  h_n_ED =  ROOT.TProfile2D("h_n_ED"+str(lay),"h_n_ED"+str(lay),64*2,-64,64,64*2,-64,64)
  ED_n_Plot.append(h_n_ED)

  h_nt_ED_rescale =  ROOT.TProfile2D("h_nt_ED_rescale"+str(lay),"h_nt_ED_rescale"+str(lay),64*2,-64,64,64*2,-64,64)
  ED_nt_Plot_rescale.append(h_nt_ED_rescale)
  h_n_ED_rescale =  ROOT.TProfile2D("h_n_ED_rescale"+str(lay),"h_n_ED_rescale"+str(lay),64*2,-64,64,64*2,-64,64)
  ED_n_Plot_rescale.append(h_n_ED_rescale)


#Main Loop
for ev in range(Events):
 pos_ = []
 ene_ = []
 posT_ = []
 eneT_ = []
 posStan_ = []
 eneStan_ = []
 if ev%50 == 0:print(ev)
 Ne_Ev=fnn["Pred_En_HighRes"][ev]
 Th_Ev=fnn["Truth_En_HighRes"][ev]
 #Redifine for truth a Non-super res 
 Standard_Res = [ [ [ 0 for i in range(64) ] for j in range(64) ] for k in range(num_L)  ]
 #cutting to select a single pT bins. We shall see the second circle reducing for higher boost. Would be nice to see...
 if np.sum(Th_Ev) < 2500 or np.sum(Th_Ev) > 3000: continue 
 for lay in range(3):
  Lay_Max_pred      = [0,0,0] 
  Lay_Max_truth     = [0,0,0]
  Lay_Max_truth_sta = [0,0,0]
  #get the maximum to shift eveything
  #also fill Standard Image
  Sup_f = 64. / layers[lay] 
  for X in range(64):
   for Y in range(64):
    New_X = int(X / Sup_f)
    New_Y = int(Y / Sup_f)
    Standard_Res[lay][New_X][New_Y] =  Standard_Res[lay][New_X][New_Y] + Th_Ev[lay][X][Y]





    if Lay_Max_pred[0]  < Ne_Ev[lay][X][Y]: Lay_Max_pred   = [Ne_Ev[lay][X][Y],X,Y]
    if Lay_Max_truth[0] < Th_Ev[lay][X][Y]: Lay_Max_truth  = [Th_Ev[lay][X][Y],X,Y]
    #Run K-mean on predicted - modify for truth later on
    if Ne_Ev[lay][X][Y] != 0 :
     pos_.append( [pos_cm(X,64), pos_cm(Y,64), Z_L[lay]/300.] )
     ene_.append( Ne_Ev[lay][X][Y]  )
    if Th_Ev[lay][X][Y] != 0 :
     posT_.append( [pos_cm(X,64), pos_cm(Y,64), Z_L[lay]/300.] )
     eneT_.append( Th_Ev[lay][X][Y]  )
    #This is to have a non super-res image



#Check this numbers

#################################################################################
##Make Average Event Display ##################################################
## TODO: add event display for single events
  #Check Maximum
  
  for X in range(layers[lay]):
   for Y in range(layers[lay]):
    if Lay_Max_truth_sta[0] < Standard_Res[lay][X][Y] : Lay_Max_truth_sta  = [Standard_Res[lay][X][Y],X,Y]


  for X in range(64):
   for Y in range(64):

     #Compute mass for standard image
     if Standard_Res[lay][X][Y] != 0:
      posStan_.append( [pos_cm(X,layers[lay]), pos_cm(Y,layers[lay]), Z_L[lay]/300.] )
      eneStan_.append( Standard_Res[lay][X][Y]  ) 
     ###############################    

     ED_n_Plot[lay].Fill(X-Lay_Max_pred[1],Y-Lay_Max_pred[2],Ne_Ev[lay][X][Y])
     ED_nt_Plot[lay].Fill(X-Lay_Max_truth[1],Y-Lay_Max_truth[2],Th_Ev[lay][X][Y])
     ED_nt_Plot_rescale[lay].Fill(X-Lay_Max_truth_sta[1],Y-Lay_Max_truth_sta[2],Standard_Res[lay][X][Y])

  #################################################################################
 reco_mass, reco_distance   = Mass_distance(pos_,ene_)
 truth_mass, truth_distance = Mass_distance(posT_,eneT_)
 truth_mass_Sta, truth_distance_Sta = Mass_distance(posStan_,eneStan_)
 h_reco_mass.Fill(reco_mass)
 h_truth_mass.Fill(truth_mass)
 h_reco_distance.Fill( reco_distance)
 h_truth_distance.Fill(truth_distance)
 h_truth_mass_sta.Fill(truth_mass_Sta)
#####TODO add non super-resoluted#####


f_o=ROOT.TFile("Epred_out_SuperResPosition.root","recreate") 
for ly in range(3):
 ED_n_Plot[ly].Write()
 ED_nt_Plot[ly].Write()
 ED_nt_Plot_rescale[ly].Write()
h_reco_mass.      Write() 
h_truth_mass.     Write() 
h_truth_mass_sta.     Write() 
h_reco_distance.  Write() 
h_truth_distance. Write()
f_o.Close()
