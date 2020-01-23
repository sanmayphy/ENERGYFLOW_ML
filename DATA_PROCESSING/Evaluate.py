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


Events = 6000

f = h5py.File('/afs/cern.ch/work/s/sanmay/public/Outfile_CellInformation_HomDet_2to5GeV_FineGran128_V1.h5','r')

gT     = h5py.File('Outfile_2to5GeV_TotalTopo.h5','r')
gEpred = h5py.File('Outfile_2to5GeV_TotalEpred.h5','r')

#h_Epred_prof_tot= ROOT.TProfile2D("h_Epred_prof_tot","h_Epred_prof_tot",5000000,1.5e3,5.5e3,6,0,6) 
h_Epred_prof_tot= ROOT.TProfile2D("h_Epred_prof_tot","h_Epred_prof_tot",50,1.5e3,5.5e3,6,0,6) 

C1=f["Trk_Theta"]
C2=f["Trk_Phi"]
C3=f["Trk_X_pos"]
C4=f["Trk_Y_pos"]


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
 En=f["Smeared_Track_Energy"][ev]
 ptr=math.sqrt(En*En-139.57018*139.57018)
 Traj_Mark=Track.MakeTruthTrajectory(C1[ev], C2[ev], C3[ev], C4[ev])
 total=[]
 TC = -1




 for lay in range(6):
      Ch_Ev=f["RealRes_ChargedEnergy_Layer"+str(lay+1)][ev][0]
      Neu_Ev=f["RealRes_NeutralEnergy_Layer"+str(lay+1)][ev][0]
      Lay_Ev=f["RealRes_TotalEnergy_Layer"+str(lay+1)][ev][0]
      #Etopo_Ev =gEpred["PflowTC"+str(lay+1)][ev][0][:layers[lay],:layers[lay]]
      Etopo_Ev =gEpred["PflowTC"+str(lay+1)][ev][0]
      Topo_Ev =gT["TopoClusters"+str(lay+1)][ev][0][:layers[lay],:layers[lay]]
      Epred_Ev =gEpred["PflowPred"+str(lay+1)][ev][0][:layers[lay],:layers[lay]]
      Epred_tot1 =gEpred["PflowPredTot"+str(lay+1)][ev][0][:layers[lay],:layers[lay]]
      Epred_LFI =gEpred["PflowLFI"+str(lay+1)][ev][0][:layers[lay],:layers[lay]]
      Noise_Ev=f["RealRes_Noise_Layer"+str(lay+1)][ev][0]
      #Ch_Ev = Ch_Ev + Noise_Ev
      #Lay_Ev = Lay_Ev - Noise_Ev
      for X in range(layers[lay]):
        for Y in range(layers[lay]):
         #this needs to be improved - changed to give which TC correpsonds to which
#         if Etopo_Ev[X][Y] != 0:
#           TC = Etopo_Ev[X][Y]
         if   Etopo_Ev[X][Y]!= 0:
           pred = Epred_tot1[X][Y]
#           if Topo_Ev[X][Y]!=Etopo_Ev[X][Y]: print "here:", Etopo_Ev[X][Y],Topo_Ev[X][Y],X,Y,lay
           #print Etopo_Ev[X][Y], Topo_Ev[X][Y]
         #if Etopo_Ev[X][Y] == Topo_Ev[X][Y]:
           lf = Epred_LFI[X][Y]
         #if TC == Topo_Ev[X][Y] and TC != 0:
           Etot = Lay_Ev[X][Y] + Etot
           Eneutral = Neu_Ev[X][Y] + Eneutral
           EPflow = Epred_Ev[X][Y] + EPflow
           Ech = Ch_Ev[X][Y] + Ech
 if Ech!=0: h_Epred_prof_tot.Fill(ptr,lf,Ech/ptr) 
 Epred_tot = ptr*hProf_t.GetBinContent(hProf_t.GetXaxis().FindBin(ptr),hProf_t.GetYaxis().FindBin(lf))
# print Epred_tot, Ech, EPflow,pred
 if  Ech!=0 and Epred_tot!= 0:
 #if Eneutral != 0 and Ech!=0 and TC != 0:
   h_Energy.Fill( Ech  ) 
   h_Eneutral.Fill( Eneutral  ) 
#   print  (Epred_tot-pred),TC 
   h_Pflow.Fill(((Etot-pred) - Eneutral)/Eneutral)   
   h_Pflow1.Fill(((Etot-EPflow) - Eneutral)/Eneutral)   
f_o=ROOT.TFile("Epred_out.root","recreate") 
h_Pflow.Write()
h_Pflow1.Write()
h_Energy.Write()
h_Eneutral.Write()
h_Epred_prof_tot.Write()
f_o.Close()
