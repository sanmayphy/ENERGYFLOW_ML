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



algorithms = [#["Pflow","PflowPred","/afs/cern.ch/work/f/fdibello/public/Outfile_2to5GeV_TotalEpred.h5"],
              #["CNN"  ,"RealRes_NeutralEnergyPred_Layer","/afs/cern.ch/work/s/sanmay/public/Outfile_MLPred_HomDet_2to5GeV_FineGran128_V1.h5"]]
              ["DeepSet"  ,"RealRes_NeutralEnergyPred_Layer","/eos/user/s/sanmay/public/MULTIPLE_TRAINING/Outfile_DeepSet_HomDet_15to20GeV_TopoCluster_WS.h5"],
              ["Graph"  ,"RealRes_NeutralEnergyPred_Layer","/eos/user/s/sanmay/public/MULTIPLE_TRAINING/Outfile_Graph_HomDet_15to20GeV_TopoCluster_WS.h5"],
              ["Unet"  ,"RealRes_NeutralEnergyPred_Layer","/eos/user/s/sanmay/public/MULTIPLE_TRAINING/Outfile_UNet_HomDet_15to20GeV_TopoCluster_WS.h5"]]

#TestFile = "/eos/user/s/sanmay/public/MULTIPLE_TRAINING/Outfile_ConvNet_HomDet_5to10GeV_TopoCluster_WS.h5"
TestFile = "/afs/cern.ch/work/s/sanmay/public/Outfile_CellInformation_HomDet_15to20GeV_FineGran128_V1.h5"
TCFile = "/eos/user/s/sanmay/public/MULTIPLE_TRAINING/TopoClusterFile_WS.h5"
#TCFile = "/afs/cern.ch/work/f/fdibello/Outfile_2to5GeV_TotalTopo.h5"
Events = 5999
PflowPred = ""


# General Config file
f      = h5py.File(TestFile,'r')
gT     = h5py.File(TCFile,'r')



#General track parameters
layers=[64,32,32,16,16,8]
print(algorithms[0][2])
gPF = h5py.File(algorithms[0][2],'r')

print("Keys: %s" % gPF.keys())

Fpred = []
Hpred = []
for algo in algorithms:
 print("Running with algo: ",algo[0]) 
 # Add here the histograms
 gEpred = h5py.File(algo[2],'r')
 Fpred.append([gEpred,algo[1]])
 #Figure of Merit used for energy distribution within the TC
 h_pred =  ROOT.TH1D("h_Energy"+algo[0],"h_Energy"+algo[0],400,-2,2)
 Hpred.append(h_pred)

 out_image = []
for ev in range(Events):
 if ev%200 == 0:print("Analyzed:", ev)
 Etot = 0
 Eneutral = 0
 Ech = 0
 #En=f["Smeared_Track_Energy"][ev]
 #ptr=math.sqrt(En*En-139.57018*139.57018)
 #Traj_Mark=Track.MakeTruthTrajectory(C1[ev], C2[ev], C3[ev], C4[ev])
 total=[]
 TC = -1
 FigureOfMerit = []
 for algo in algorithms:
  FigureOfMerit.append(0)

#This needs to be fixed
# for lay in range(6):
#    Etopo_t =gPF["PflowTC"+str(lay+1)][ev][0][:layers[lay],:layers[lay]]
#    for X in range(layers[lay]):
#      for Y in range(layers[lay]):
#       if Etopo_t[X][Y] != 0: 
#        TC = Etopo_t[X][Y] 
#        break 
# This is nasty and hard coded
 TC = 1
 # This is the total energy of each prediction per event
 E = []
 for lay in range(6):
     #Traj_Mark[lay][0]= Traj_Mark[lay][0] -1 
     #Traj_Mark[lay][1]= Traj_Mark[lay][1] -1 
     #Ch_Ev=f["RealRes_ChargedEnergy_Layer"+str(lay+1)][ev][0][:layers[lay],:layers[lay]]
     Neu_Ev=gEpred["RealRes_NeutralEnergyTruth_Layer"+str(lay+1)][ev][0][:layers[lay],:layers[lay]]
     Lay_Ev=gEpred["RealRes_TotalEnergy_Layer"+str(lay+1)][ev][0][:layers[lay],:layers[lay]]
     Topo_Ev =gT["topocluster_15to20"][ev][lay:lay+1,:layers[lay],:layers[lay]]
     Ch_Ev = Lay_Ev - Neu_Ev
     Epred = []
     for pred in Fpred:
      Epred_Ev =pred[0][pred[1]+str(lay+1)][ev][0][:layers[lay],:layers[lay]]
      Epred.append(Epred_Ev)
     for X in range(layers[lay]):
       for Y in range(layers[lay]):
        if Topo_Ev[0][X][Y] >  0 and TC != 0:
        #if Topo_Ev[0][X][Y] == TC and TC != 0:
          Etot = Lay_Ev[X][Y] + Etot
          Eneutral = Neu_Ev[X][Y] + Eneutral
          Ech = Ch_Ev[X][Y] + Ech
          a = -1
          for pre in Epred:
           a = a + 1 
           FigureOfMerit[a] = pre[X][Y] + FigureOfMerit[a]
 #Check to eliminate funny cases, these are typcally 0.3 % of the dataset
 i = -1
 for pre in Epred:
  i = i + 1
  if  Ech!=0 and FigureOfMerit[i]!= 0 and Etot != 0 and Eneutral != 0 and int(FigureOfMerit[i]) != int(Etot):
   #Very coarse wayif i == 0: Hpred[i].Fill(((Etot-FigureOfMerit[i]) - Eneutral)/Eneutral)   
   if algo[0] == "Pflow": Hpred[i].Fill(((Etot-FigureOfMerit[i]) - Eneutral)/Eneutral)  
   else: Hpred[i].Fill(((FigureOfMerit[i]) - Eneutral)/Eneutral)   
f_o=ROOT.TFile("Epred_outALL.root","recreate") 
for h_pred in Hpred:
 h_pred.Write()
f_o.Close()
