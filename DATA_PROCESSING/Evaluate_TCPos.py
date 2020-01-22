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


Events = 1000

f = h5py.File('/afs/cern.ch/work/s/sanmay/public/Outfile_CellInformation_HomDet_2to5GeV_FineGran128_V1.h5','r')
fnn = h5py.File('/afs/cern.ch/work/s/sanmay/public/Outfile_MLPred_HomDet_2to5GeV_FineGran128_V1.h5','r')

gT     = h5py.File('Outfile_2to5GeV_TotalTopo.h5','r')
gTnn     = h5py.File('Outfile_2to5GeV_TotalTopo.h5','r')
#gT     = h5py.File('Outfile_2to5GeV_ChargedTopo.h5','r')
gEpred = h5py.File('Outfile_2to5GeV_TotalEpred.h5','r')

#h_Epred_prof_tot= ROOT.TProfile2D("h_Epred_prof_tot","h_Epred_prof_tot",5000000,1.5e3,5.5e3,6,0,6) 
h_Epred_prof_tot= ROOT.TProfile2D("h_Epred_prof_tot","h_Epred_prof_tot",50,1.5e3,5.5e3,6,0,6) 

C1=f["Trk_Theta"]
C2=f["Trk_Phi"]
C3=f["Trk_X_pos"]
C4=f["Trk_Y_pos"]

dR_n_Plot = []
dR_p_Plot = []
dR_t_Plot = []
ED_n_Plot = []
ED_p_Plot = []
ED_pN_Plot = []
ED_t_Plot = []
ED_tot_Plot = []
ED_ch_Plot = []
for lay in range(6):
  print lay
  h_n_dR =  ROOT.TH1D("h_n_dR"+str(lay),"h_n_dR"+str(lay),100,-5,80)
  dR_n_Plot.append(h_n_dR)
  h_p_dR =  ROOT.TH1D("h_p_dR"+str(lay),"h_p_dR"+str(lay),100,-5,80)
  dR_p_Plot.append(h_p_dR)
  h_t_dR =  ROOT.TH1D("h_t_dR"+str(lay),"h_t_dR"+str(lay),100,-5,80)
  dR_t_Plot.append(h_t_dR)


  h_n_ED =  ROOT.TProfile2D("h_n_ED"+str(lay),"h_n_ED"+str(lay),64,-32,32,64,-32,32)
  ED_n_Plot.append(h_n_ED)
  h_p_ED =  ROOT.TProfile2D("h_p_ED"+str(lay),"h_p_ED"+str(lay),64,-32,32,64,-32,32)
  ED_p_Plot.append(h_p_ED)
  h_pN_ED =  ROOT.TProfile2D("h_pN_ED"+str(lay),"h_pN_ED"+str(lay),64,-32,32,64,-32,32)
  ED_pN_Plot.append(h_pN_ED)
  h_t_ED =  ROOT.TProfile2D("h_t_ED"+str(lay),"h_t_ED"+str(lay),64,-32,32,64,-32,32)
  ED_t_Plot.append(h_t_ED)
  h_ch_ED =  ROOT.TProfile2D("h_ch_ED"+str(lay),"h_ch_ED"+str(lay),64,-32,32,64,-32,32)
  ED_ch_Plot.append(h_ch_ED)
  h_tot_ED =  ROOT.TProfile2D("h_tot_ED"+str(lay),"h_tot_ED"+str(lay),64,-32,32,64,-32,32)
  ED_tot_Plot.append(h_tot_ED)




h_Pflow =  ROOT.TH1D("h_Pflow","h_Pflow",50,-2,2)
h_Pflow1 =  ROOT.TH1D("h_Pflownew","h_Pflownew",50,-2,2)
h_Energy =  ROOT.TH1D("h_Energy","h_Energy",200,-500,10000)
h_Eneutral =  ROOT.TH1D("h_Eneutral","h_Eneutral",200,-500,10000)

f1 = ROOT.TFile.Open("Epred_outN.root")
#f1 = ROOT.TFile.Open("fpred_new_final.root")
hProf_t = f1.Get("h_Epred_prof_tot")
layers=[64,32,32,16,16,8]
out_image = []
for ev in range(Events):
 if ev%200 == 0:print ev
 EPflow = 0
 Etot = 0
 Eneutral = 0
 Ech = 0
 Enn = 0
 En=f["Smeared_Track_Energy"][ev]
 ptr=math.sqrt(En*En-139.57018*139.57018)
 Traj_Mark=Track.MakeTruthTrajectory(C1[ev], C2[ev], C3[ev], C4[ev])
 total=[]
 TC = -1




 for lay in range(6):
      Ch_Ev=f["RealRes_ChargedEnergy_Layer"+str(lay+1)][ev][0]
      Neu_Ev=f["RealRes_NeutralEnergy_Layer"+str(lay+1)][ev][0]
      Lay_Ev=f["RealRes_TotalEnergy_Layer"+str(lay+1)][ev][0]
      ENN_Ev=fnn["RealRes_NeutralEnergyPred_Layer"+str(lay+1)][ev][0]
      #Etopo_Ev =gEpred["PflowTC"+str(lay+1)][ev][0][:layers[lay],:layers[lay]]
      Etopo_Ev =gEpred["PflowTC"+str(lay+1)][ev][0]
      Topo_Ev =gT["TopoClusters"+str(lay+1)][ev][0][:layers[lay],:layers[lay]]
      TopoNN_Ev =gTnn["TopoClusters"+str(lay+1)][ev][0][:layers[lay],:layers[lay]]
      Epred_Ev =gEpred["PflowPred"+str(lay+1)][ev][0][:layers[lay],:layers[lay]]
      Epred_tot1 =gEpred["PflowPredTot"+str(lay+1)][ev][0][:layers[lay],:layers[lay]]
      Epred_LFI =gEpred["PflowLFI"+str(lay+1)][ev][0][:layers[lay],:layers[lay]]
      Noise_Ev=f["RealRes_Noise_Layer"+str(lay+1)][ev][0]
      #Ch_Ev = Ch_Ev + Noise_Ev
#      Lay_Ev = Lay_Ev - Noise_Ev
      distanceP = 0
      distanceN = 0
      distanceT = 0
      xAp = 0
      yAp = 0
      xAn = 0
      yAn = 0
      xAt = 0
      yAt = 0
      
      for X in range(layers[lay]):
        for Y in range(layers[lay]):
         #Truth
         Eneutral = Neu_Ev[X][Y] + Eneutral
         xAt = X* Neu_Ev[X][Y] + xAt
         yAt = Y* Neu_Ev[X][Y] + yAt
         ED_t_Plot[lay].Fill(X-Traj_Mark[lay][0],Y-Traj_Mark[lay][1],Neu_Ev[X][Y])
         if    Topo_Ev[X][Y] != 0:
         #if   Etopo_Ev[X][Y]!= 0:
           pred = Epred_tot1[X][Y]
           lf = Epred_LFI[X][Y]
         #if TC == Topo_Ev[X][Y] and TC != 0:
           Etot = Lay_Ev[X][Y] + Etot
           EPflow = Epred_Ev[X][Y] + EPflow
           Ech = Ch_Ev[X][Y] + Ech
           Enn = ENN_Ev[X][Y] + Enn
           #Pflow
           xAp = X* (Lay_Ev[X][Y]-Epred_Ev[X][Y]) + xAp
           yAp = Y* (Lay_Ev[X][Y]-Epred_Ev[X][Y]) + yAp
           ED_p_Plot[lay].Fill(X-Traj_Mark[lay][0],Y-Traj_Mark[lay][1],Epred_Ev[X][Y])
           ED_pN_Plot[lay].Fill(X-Traj_Mark[lay][0],Y-Traj_Mark[lay][1],Lay_Ev[X][Y]-Epred_Ev[X][Y])
           ED_ch_Plot[lay].Fill(X-Traj_Mark[lay][0],Y-Traj_Mark[lay][1],Ch_Ev[X][Y])
           ED_tot_Plot[lay].Fill(X-Traj_Mark[lay][0],Y-Traj_Mark[lay][1],Lay_Ev[X][Y])

         if TopoNN_Ev[X][Y] != 0: 
           #NN
           xAn = X* ENN_Ev[X][Y] + xAn
           yAn = Y* ENN_Ev[X][Y] + yAn
           ED_n_Plot[lay].Fill(X-Traj_Mark[lay][0],Y-Traj_Mark[lay][1],ENN_Ev[X][Y])



      if EPflow == 0 or Eneutral == 0 or Enn == 0: continue
     
      xAp = xAp / (Etot-EPflow)
      xAt = xAt / Eneutral
      xAn = xAn / Enn

      yAp = yAp / (Etot-EPflow)
      yAt = yAt / Eneutral
      yAn = yAn / Enn

 
      dr = np.sqrt((xAp-xAt)*(xAp-xAt)+(yAp-yAt)*(yAp-yAt))
      distanceP = distanceP + dr
#      print distanceP, dr
#      distanceP = distanceP / (Etot-EPflow)
      dr = np.sqrt((xAn-xAt)*(xAn-xAt)+(yAn-yAt)*(yAn-yAt))
      distanceN = distanceN + dr
 #     distanceN = distanceN / Enn
      dr = np.sqrt(xAt*xAt+yAt*yAt)
      distanceT = distanceT + dr
 #     distanceT = distanceT / Eneutral
 


      dR_p_Plot[lay].Fill(distanceP) 
      dR_n_Plot[lay].Fill(distanceN) 
      dR_t_Plot[lay].Fill(distanceT) 




f_o=ROOT.TFile("Epred_out_Position.root","recreate") 
for ly in range(6):
 dR_p_Plot[ly].Write()
 dR_n_Plot[ly].Write()
 ED_p_Plot[ly].Write()
 ED_pN_Plot[ly].Write()
 ED_n_Plot[ly].Write()
 ED_t_Plot[ly].Write()
 ED_tot_Plot[ly].Write()
 ED_ch_Plot[ly].Write()
f_o.Close()
