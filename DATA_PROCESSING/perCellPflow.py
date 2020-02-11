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
from operator import itemgetter

h_Pflow =  ROOT.TH1D("h_Pflow","h_Pflow",50,-1,1)
f = h5py.File('/afs/cern.ch/work/s/sanmay/public/Outfile_CellInformation_HomDet_5to10GeV_FineGran128_V2.h5')
#f = h5py.File('/afs/cern.ch/work/s/sanmay/public/Outfile_CellInformation_HomDet_2to5GeV_FineGran128_V1.h5','r')

#gT = h5py.File('/afs/cern.ch/user/l/losanti/public/Outfile_2to5GeV_ChargedTopo.h5','r')
#gT = h5py.File('Outfile_10to15GeV_ChargedTopo.h5','r')
gT = h5py.File('Outfile_2to5GeV_TotalTopo.h5','r')
#gT = h5py.File('Outfile_2to5GeV_ChargedTopo.h5','r')

Events = 6000

histo_fpred = []
h_Epred_prof_tot= ROOT.TProfile2D("h_Epred_prof","h_Epred_prof",130,1.5e3,15.5e3,6,0,6) 
for lfi in range(6):
 for l in range(6):
  h_Epred_prof= ROOT.TProfile3D("h_Epred_prof"+str(l)+str(lfi),"h_Epred_prof"+str(l)+str(lfi),75,1.5e3,15.5e3,6,0,6,60,-10,10) 
  histo_fpred.append([h_Epred_prof])

C1=f["Trk_Theta"]    
C2=f["Trk_Phi"]
C3=f["Trk_X_pos"]   
C4=f["Trk_Y_pos"]

ED_p_ED = []
ED_pt_ED = []
for lay in range(6):
  print lay
  h_p_ED =  ROOT.TProfile2D("h_p_ED"+str(lay),"h_p_ED"+str(lay),64,-32.5,31.5,64,-32.5,31.5)
  ED_p_ED.append(h_p_ED)
  h_pt_ED =  ROOT.TProfile2D("h_pt_ED"+str(lay),"h_pt_ED"+str(lay),64,-32.5,31.5,64,-32.5,31.5)
  ED_pt_ED.append(h_pt_ED)


layers=[64,32,32,16,16,8]
f1 = ROOT.TFile.Open("fpred_new_final.root")
out_image = []
out_image1 = []
out_image2 = []
out_image3 = []
for ev in range(Events):
 TC = -1
 En=f["Smeared_Track_Energy"][ev]
 ptr=math.sqrt(En*En-139.57018*139.57018)
 Traj_Mark=Track.MakeTruthTrajectory(C1[ev], C2[ev], C3[ev], C4[ev]) 
 total=[]
 for lay in range(6):
      Traj_Mark[lay][0]= Traj_Mark[lay][0] -1 
      Traj_Mark[lay][1]= Traj_Mark[lay][1] -1 
#      Lay_Ev=f["RealRes_ChargedEnergy_Layer"+str(lay+1)][ev][0]
      Lay_Ev=f["RealRes_TotalEnergy_Layer"+str(lay+1)][ev][0]
#      Noise_Ev=f["RealRes_Noise_Layer"+str(lay+1)][ev][0]
      Lay_Ev = Lay_Ev + Noise_Ev
      Topo_Ev =gT["TopoClusters"+str(lay+1)][ev][0][:layers[lay],:layers[lay]]
      for X in range(layers[lay]):
        for Y in range(layers[lay]):
         if Topo_Ev[X][Y] != 0:
          total.append([lay,X,Y,Lay_Ev[X][Y],Topo_Ev[X][Y],np.sqrt((X-Traj_Mark[lay][0])*(X-Traj_Mark[lay][0]) + (Y-Traj_Mark[lay][1])*(Y-Traj_Mark[lay][1])    )]) 
 if total == []: total.append([0,0,0,0,0])
 LFI = Track.ComputeLFI(total,Traj_Mark)
 max_value = max(total, key=itemgetter(4))
 dist = 0
 mx = int(max_value[4])
 en = []
 closest = []
 for m in range(1,mx+1):
  average_X =  0  
  average_Y =  0 
  energy = 0 
  tot =  0 
  totA =  0
  for cell in total:
   if m != cell[4]: continue 
   if cell[0] == LFI:
    average_X = average_X + cell[1]
    average_Y = average_Y + cell[2]
    tot = tot + 1.
   totA = totA + 1.
   energy = energy + cell[3]
  if tot != 0:
   average_X = average_X/tot
   average_Y = average_Y/tot
  else:
   average_X = 10000
   average_Y = 10000
  dR = np.sqrt((average_X-Traj_Mark[LFI][0])*(average_X-Traj_Mark[LFI][0]) + (average_Y-Traj_Mark[LFI][1])*(average_Y-Traj_Mark[LFI][1])) 
  en.append([m,energy])
  closest.append([m,dR,tot,totA])
       
 
 en.sort(reverse = True,key=lambda en: en[1])
 closest.sort(reverse = False,key=lambda closest: closest[1])
 #TC = en[0][0]  # this is the number of the TC corresponding to the one associated to the track 
 if closest != []: TC = closest[0][0]  # this is the number of the TC corresponding to the one associated to the track 
# now that the number of the most eergetic TC is found, it is possible to compute Epred and run the concentric circle stuff
# print TC,LFI
# now open up the circles 
 #print closest 


 #f1 = ROOT.TFile.Open("~/fpred.root")
 #print Traj_Mark[0][0], Traj_Mark[1][0]
 hProf_t = f1.Get("h_Epred_prof")
 Epred_tot = ptr*hProf_t.GetBinContent(hProf_t.GetXaxis().FindBin(ptr),hProf_t.GetYaxis().FindBin(LFI)) 

 # now open up the circles
 layer_c = []
 total_en = np.zeros((100,6))
 total_enA = np.zeros((100,6))
 total_enP = np.zeros((100,6))
 total_enC = np.zeros((100,6))
 dist = 1.2
 Energia_total = 0
 for cell in total:
  if cell[4]!= TC: continue
  Energia_total = Energia_total + cell[3]

 if Energia_total != 0:h_Epred_prof_tot.Fill(ptr,LFI,Energia_total/ptr)
 for cell in total:
  if cell[4] != TC: continue
  ED_pt_ED[cell[0]].Fill(cell[1]-Traj_Mark[cell[0]][0],cell[2]-Traj_Mark[cell[0]][1],cell[3])
  histo_fpred[6*LFI+cell[0]][0].Fill(ptr,LFI+0.1,cell[5]+0.1,cell[3]/ptr)
#  print "LFI ",LFI, "lay = ",cell[0]
#  print histo_fpred[6*LFI+cell[0]][0].GetName()

  hProf = f1.Get("h_Epred_prof"+str(cell[0])+str(LFI))
 # print "h_Epred_prof"+str(LFI)+str(cell[0])
  ePred = ptr*hProf.GetBinContent(hProf.GetXaxis().FindBin(ptr),hProf.GetYaxis().FindBin(LFI+0.1),hProf.GetZaxis().FindBin(cell[5]+0.1))
  #print ePred, cell[3],cell[5]," layer = ",cell[0]
  
  for circle in range(0,100):
   ell = 0 
   if cell[5] <= circle*dist and cell[5] > (circle-1)*dist: 
    if circle == 0: 
     ell = (3.14)
     layer_c.append(cell+[circle,cell[3]/(3.14),cell[3]])
    elif circle == 1: 
     ell = (-3.14+3.14*circle*dist*circle*dist)
     layer_c.append(cell+[circle,cell[3]/(-3.14+3.14*circle*dist*circle*dist),cell[3]])
    else: 
     ell = ((3.14*circle*dist*circle*dist)-(3.14*(circle-1)*dist*(circle-1)*dist))
     layer_c.append(cell+[circle,cell[3]/ell,cell[3]])
    #print ell, circle,cell[0],cell[5] 
    total_en[circle][cell[0]] = total_en[circle][cell[0]]  + cell[3]
    total_enA[circle][cell[0]] = total_enA[circle][cell[0]]  + (cell[3]/ell) * ePred
    #total_enA[circle][cell[0]] = total_enA[circle][cell[0]]  + cell[3]/ell
    total_enP[circle][cell[0]] = total_enP[circle][cell[0]]  + ePred
    total_enC[circle][cell[0]] = total_enC[circle][cell[0]]  + 1
    break  

 decor = []
 totE = 0
 totEa = 0
 totEp = 0
 totEc = 0
 for cly in layer_c:
  totE =  total_en[cly[6]][cly[0]]
  totEa =  total_enA[cly[6]][cly[0]]
  totEp =  total_enP[cly[6]][cly[0]]
  totEc =  total_enC[cly[6]][cly[0]]
  decor.append(cly+[totE,totEa,totEp,totEc])
  #print totE, totEp, LFI,cly[6],cly[0]
 #now remove the topoclustering
 
# decor.sort(reverse = False,key=lambda decor: decor[10])
 decor.sort(reverse = True,key=lambda decor: decor[10])
 Epr = 0
 app = -1
 cell_tot = []
 decor1 = decor
 Epred_sum = 0
 for final in decor:
  if final[9] != app:
   Epr =  final[9]
   #Epr = Epr +  final[9]
   app = final[9] 
   ePred = final[9]
   #ePred = final[11]
   Epred_sum = ePred + Epred_sum
   celln = -1
   for fn in decor1:
    celln = celln + 1
    if fn[0] == final[0] and fn[6] == final[6]: 
       
      if Epred_sum <= Epred_tot: 
       cell_tot.append([fn[0],fn[1],fn[2],ePred/final[12],TC])
       ED_p_ED[fn[0]].Fill(fn[1]-Traj_Mark[fn[0]][0],fn[2]-Traj_Mark[fn[0]][1],ePred/final[12])
   #   if Epred_sum <= Epred_tot: cell_tot.append([fn[0],fn[1],fn[2], 0.000001])
      elif Epred_sum > Epred_tot:
       #print final[12], Epred_sum/Epred_tot, "+++++++++" 
       cell_tot.append([fn[0],fn[1],fn[2], ePred/final[12]* (Epred_tot/Epred_sum) ,TC])
       ED_p_ED[fn[0]].Fill(fn[1]-Traj_Mark[fn[0]][0],fn[2]-Traj_Mark[fn[0]][1],ePred/final[12]* (Epred_tot/Epred_sum))
       #cell_tot.append([fn[0],fn[1],fn[2], ((Epred_sum-Epred_tot))/final[12],TC])
   if Epred_sum >= Epred_tot: break
 rescale = 1
 if Epred_sum> 0: rescale = Epred_tot/Epred_sum
# print Epred_sum, Epred_tot, Epred_tot/Epred_sum 
      #print "minore **", Epred_sum, Epred_tot,rescale,ev,final[12],final[3]
 #if Epred_sum < Epred_tot:print "minore ", Epred_sum, Epred_tot,rescale
 #if rescale > 1: print final 
 #need to rescale up the missing energy
 cell_tot_res = []
 for cll in cell_tot:
  if rescale <= 1: cell_tot_res.append(cll)
  else: cell_tot_res.append([cll[0],cll[1],cll[2],cll[3]*rescale,TC])

  

      #if fn[3] != 0: 
      # h_Pflow.Fill((ePred/final[12]-fn[3])/fn[3])
 #print cell_tot 
 #print "*******results:", Epr,ePred,Epr - ePred 
 if ev%200 == 0:print ev
# h = Track.Assign_Epred(total)
 h, h1 = Track.Assign_Epred(cell_tot)
 #h = Track.Assign_Epred(cell_tot_res)
 out_image.append(h)

# h1 = Track.Assign_TC(total,TC,True)
 out_image1.append(h1)


 h2 = Track.Assign_TC(total,Epred_tot,False)
 out_image2.append(h2)

 h3 = Track.Assign_TC(total,LFI,False)
 out_image3.append(h3)


out_image = np.array(out_image)
out_image1 = np.array(out_image1)
out_image2 = np.array(out_image2)
out_image3 = np.array(out_image3)
print('opening File')
#
print out_image1.shape
with h5py.File('Outfile_2to5GeV_'+'Total'+'Epred.h5', 'w') as f2:
#
    for layer_i in range(6):
        f2.create_dataset('PflowPred' + str(layer_i+1), data=out_image[:, layer_i:layer_i+1,:, :])
        f2.create_dataset('PflowTC' + str(layer_i+1), data=out_image1[:, layer_i:layer_i+1,:, :])
        f2.create_dataset('PflowPredTot' + str(layer_i+1), data=out_image2[:, layer_i:layer_i+1,:, :])
        f2.create_dataset('PflowLFI' + str(layer_i+1), data=out_image3[:, layer_i:layer_i+1,:, :])

print ("Exiting ... Bye!")
#
fpred_new = ROOT.TFile("fpred_new.root","recreate")
for h in histo_fpred:
 h[0].Write()
ED_p_ED[0].Write()
ED_p_ED[1].Write()
ED_p_ED[2].Write()
ED_pt_ED[0].Write()
ED_pt_ED[1].Write()
ED_pt_ED[2].Write()
h_Epred_prof_tot.Write()
h_Pflow.Write()
fpred_new.Close()
