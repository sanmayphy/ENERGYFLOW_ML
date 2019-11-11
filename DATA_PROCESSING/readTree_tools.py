import ROOT
from sys import exit
import random
from array import array
import numpy as np
import EventDisplay
import tools
#import numpy as np
#from root_numpy import tree2array,array2tree, array2root
ROOT.gROOT.SetBatch(True) 


def Matching(p,track):
    p_match = []
    p_start = []
    delta_start = 1e10
    sigmaX = []   
    sigmaY = [] 
    for pM in p: 
     sigmaX = []   
     sigmaY = [] 
     e = 0 
     for cell in pM:
      sigmaX.append(cell[2])  
      sigmaY.append(cell[1]) 
      e = cell[3] + e
     Sx= np.std(sigmaX)
     Sy= np.std(sigmaY)
     Mx= np.mean(sigmaX)
     My= np.mean(sigmaY)
     DeltaR = ((track[0][2]-Mx)/Sx) * ((track[0][2]-Mx)/Sx)  + ((track[0][1]-My)/Sy)*((track[0][1]-My)/Sy)
     DeltaR = np.sqrt(DeltaR)
     if track[0][3] > 0:
      if DeltaR < delta_start and e/track[0][3] > 0.1: 
       delta_start = DeltaR
       p_start = pM
    print "cluster positions X:",sigmaX
    print "cluster positions Y:",sigmaY
    p_match.append(p_start)
    print "matching cluster delta is ",delta_start
    return delta_start,p_match




def AttachEnergy(proto_final,L,Ch,Nu):
    p_withE = []
    for p in proto_final:
     c_E = []
     for c in p:
      energy=[]
      j = c[2] + 64*c[1] + 4096*c[0]
      energy.append(L[j])
      energy.append(Ch[j])
      energy.append(Nu[j])
      c_E.append(c + energy)
     p_withE.append(c_E)
    return p_withE

def GetTrackParam(proto):
    track = []
    p2 = []
    p2_2 = []
    for p1 in proto:
     if len(p2) < p1: p2 = p1
    #I have merged all the topo cluster into one single to estimate track energy
    p2_2.append(p2)
    if p2 == []: print "no charged cluster formed"
    z = 0
    x = 0
    for p2 in proto:
     x = 0
     y = 0
     e = 0
     lay = 0
     cells = 0
     z = 0
     for c in p2:
      cells = cells + c[3] 
      x = x + c[2]*c[3]
      y = y + c[1]*c[3]
      e = e + c[3]
     x = float(x) / float(cells)
     y = float(y)/ float(cells)
     e = e
    if x == 0: return 0
    t = [z,y,x,e]
    track.append(t)
    return track

#if a cluster is close to another one with one seed then merge
def MergeS(proto,seed):
   p_fcheck = []
   p_f = []
   for p in proto:
    for p1 in proto:
     if p1 != p and p not in p_fcheck and p1 not in p_fcheck:
      cell1 = []
      for c in p1:
       cell1.append([c[0]+1,c[1],c[2]]) 
       cell1.append([c[0]-1,c[1],c[2]])
       cell1.append([c[0],c[1]+1,c[2]])
       cell1.append([c[0],c[1]-1,c[2]])
       cell1.append([c[0],c[1],c[2]+1])
       cell1.append([c[0],c[1],c[2]-1])
      for neib in cell1:
       if neib not in p and neib in seed and neib :
        p_f.append(p + p1)
        p_fcheck.append(p)
        p_fcheck.append(p1)
        break
   return p_f

def clustering(Energies,rand):
    j = -1
    S = 5.0
    N = 2.0
    P = 0.0
    seeds = []
    seedsN = []
    seedsP = []
    #making collections of cluster with significances over thresholds
    for Ecell in  Energies:
       j = j + 1
       layer = int(j/(64*64))
       cellY = int((j-layer*64*64)/64.)
       cellX =  (j-layer*64*64) % 64
       noise = 10.
       if layer == 0: noise = 10.
       if layer == 1: noise = 20.
       if layer == 2: noise = 15.
       if layer > 2: noise = 20.
      

       #En = Ecell
       if rand: En = random.gauss(0,noise) + Ecell
       else: En = Ecell
       if  En/noise > S:
         seed = [layer,cellY,cellX] 
         seeds.append(seed)
         #print "found Sseed with significance = ",Ecell/noise, seed
       if  En/noise > N and En/noise < S:
         seedN = [layer,cellY,cellX] 
         seedsN.append(seedN)
         #print "found Nseed with significance = ",Ecell/noise, seedN
       if  En/noise > P and En/noise < N:
         seedP = [layer,cellY,cellX] 
         seedsP.append(seedP)
         #print "found Pseed with significance = ",Ecell/noise, seedP
    protoCluster = []
    PC = Proto(seeds,seedsN,seedsP)
    PCm = MergeS(PC,seeds) 
 
#    print "TheProto: ",PC
    return PCm

def common_cluster(proto1,proto2):
    protoO = proto1
    for cel in proto1:
     if cel in proto2:
#     for cel2 in proto2:
#      if cel == cel2:
       protoO = proto1+proto2
       break
    return protoO,proto1,proto2

def clean_duplicates(merge_final):
   cleaned_m = []
   for m in merge_final:
     cleaned = []
     for c in m:
      if c not in cleaned:
       cleaned.append(c)
     cleaned_m.append(cleaned)
   return cleaned_m

#define here the clustering algorithm itself
def Proto(sedS,sedN,sedP):
    proto_ensemble = []
    for cell in sedS:
     proto = []
     cell_proto = []
     lay = cell[0]
     cX  = cell[1]  
     cY  = cell[2]
     proto.append(cell)
#     print "increasing the protoCluster with seed",cell
     #define the one close to it
     cell_proto.append([lay+1,cX,cY])
     cell_proto.append([lay-1,cX,cY])
     cell_proto.append([lay,cX+1,cY])
     cell_proto.append([lay,cX-1,cY])
     cell_proto.append([lay,cX,cY+1])
     cell_proto.append([lay,cX,cY-1])
     cell_proto.append([lay,cX+2,cY])
     cell_proto.append([lay,cX-2,cY])
     cell_proto.append([lay,cX,cY+2])
     cell_proto.append([lay,cX,cY-2])
     cell_proto.append([lay,cX+1,cY+1])
     cell_proto.append([lay,cX+1,cY-1])
     cell_proto.append([lay,cX-1,cY+1])
     cell_proto.append([lay,cX-1,cY-1])
     #first cluster the low energy ones
     for cp in cell_proto:
      if cp  in sedP: 
#       print "increasing the protoCluster with seed",cell," and new cell P:",cp
       proto.append(cp)     
     for cpN in cell_proto:
      if cpN in sedN: 
#       print "increasing the protoCluster with seed",cell," and new cell N:",cp
       proto.append(cpN)     
       layNP = cpN[0]
       cXNP  = cpN[1]  
       cYNP  = cpN[2]
       new_CellP = []
       new_CellP.append([layNP+1,cXNP,cYNP])
       new_CellP.append([layNP-1,cXNP,cYNP])
       new_CellP.append([layNP,cXNP+1,cYNP])
       new_CellP.append([layNP,cXNP-1,cYNP])
       new_CellP.append([layNP,cXNP,cYNP+1])
       new_CellP.append([layNP,cXNP,cYNP-1])
       new_CellP.append([layNP,cXNP+2,cYNP])
       new_CellP.append([layNP,cXNP-2,cYNP])
       new_CellP.append([layNP,cXNP,cYNP+2])
       new_CellP.append([layNP,cXNP,cYNP-2])
       new_CellP.append([layNP,cXNP+1,cYNP+1])
       new_CellP.append([layNP,cXNP+1,cYNP-1])
       new_CellP.append([layNP,cXNP-1,cYNP+1])
       new_CellP.append([layNP,cXNP-1,cYNP-1])
       for new_cell in new_CellP: 
        if new_cell not in proto:
         if new_cell in sedP:
          proto.append(new_cell)
     
     proto_ensemble.append(proto)
     proto_ensembleMerged = []
     #need now to merge
    merge_final = [] 
    merge_final_check = [] 
    i = 0
    f = 0
    for pe in proto_ensemble:
     i = i +1
     merged = pe  
     j = 0
     if pe not in merge_final_check:
      for pe_1 in proto_ensemble:
       j = j +1
       if pe_1 != pe and pe_1 not in merge_final_check:
        merged,m_app1,m_app2 = common_cluster(merged,pe_1)
        if merged != m_app1:
         merge_final_check.append(pe)
         merge_final_check.append(pe_1)
      if merged != []: f = f + 1
      if merged != []: merge_final.append(merged) 
      
        
#        merged = merged + common_cluster(pe,pe_1)
#      proto_ensembleMerged.append(merged)
#
    merge_clean = clean_duplicates(merge_final)
    return merge_clean

def computeTruth_E(Energy):
    j = -1
    Etot = [0,0,0,0,0,0]
    for Ecell in Energy:
     j = j + 1
     layer = int(j/(64*64))
     cellY = int((j-layer*64*64)/64.)
     cellX =  (j-layer*64*64) % 64
     Etot[layer] =  Etot[layer] + Ecell



    h_E_pred0.Fill(Etot[0])
    h_E_pred1.Fill(Etot[1])
    h_E_pred2.Fill(Etot[2])
    h_E_pred3.Fill(Etot[3])
    h_E_pred4.Fill(Etot[4])
    h_E_pred5.Fill(Etot[5])
    



f = ROOT.TFile.Open("/eos/user/f/fdibello/PFlowNtupleFile_DiffPos.root")

i = -1 

tree = f.Get('EventTree')

#array = tree2array(tree,branches=['cellNu_Energy'], selection='', start=0, stop=1000, step=1)
#print array.shape
def Histogramming(Proto,label):
   h_averageProtoL0= ROOT.TH2D("h_averageProtoL0"+label,"h_averageProtoL0"+label,64,0,64,64,0,64)
   h_averageProtoL1= ROOT.TH2D("h_averageProtoL1"+label,"h_averageProtoL1"+label,64,0,64,64,0,64)
   h_averageProtoL2= ROOT.TH2D("h_averageProtoL2"+label,"h_averageProtoL2"+label,64,0,64,64,0,64)
   h_averageProtoL3= ROOT.TH2D("h_averageProtoL3"+label,"h_averageProtoL3"+label,64,0,64,64,0,64)
   h_averageProtoL4= ROOT.TH2D("h_averageProtoL4"+label,"h_averageProtoL4"+label,64,0,64,64,0,64)
   h_averageProtoL5= ROOT.TH2D("h_averageProtoL5"+label,"h_averageProtoL5"+label,64,0,64,64,0,64)
   nEns = 0
   nCell = 0
   for cell_proto in Proto:
    nCell = 0
    nEns = nEns + 1
    for cell in cell_proto:
     nCell = nCell + 1
     if "total" in label : nEns = cell[3]
     elif "charged" in label: nEns = cell[4] 
     elif "neutral" in label: nEns = cell[5]
     elif "fraction" in label:  
      if cell[3]!=0: nEns = cell[4]/cell[3]
      else: nEns = 0.
     else: nEns = nEns 
     h_averageProto.SetBinContent(cell[0],cell[1],cell[2],nEns)
     if cell[0] == 0: h_averageProtoL0.SetBinContent(cell[1],cell[2],nEns)
     if cell[0] == 1: h_averageProtoL1.SetBinContent(cell[1],cell[2],nEns)
     if cell[0] == 2: h_averageProtoL2.SetBinContent(cell[1],cell[2],nEns)
     if cell[0] == 3: h_averageProtoL3.SetBinContent(cell[1],cell[2],nEns)
     if cell[0] == 4: h_averageProtoL4.SetBinContent(cell[1],cell[2],nEns)
     if cell[0] == 5: h_averageProtoL5.SetBinContent(cell[1],cell[2],nEns)
   if "total" in label: hCS_total.Fill(nCell)  
   if "Topo" in label: print "number of topocluster:", nEns
   h_averageProtoL0.Write()
   h_averageProtoL1.Write()
   h_averageProtoL2.Write()
   h_averageProtoL3.Write()
   h_averageProtoL4.Write()
   h_averageProtoL5.Write()

def histos(SS,NN,PP):
    for cell in SS:
     if cell[0] == 0: h_averageProtoL0_S.SetBinContent(cell[1],cell[2],1)
     if cell[0] == 1: h_averageProtoL1_S.SetBinContent(cell[1],cell[2],1)
     if cell[0] == 2: h_averageProtoL2_S.SetBinContent(cell[1],cell[2],1)
     if cell[0] == 3: h_averageProtoL3_S.SetBinContent(cell[1],cell[2],1)
     if cell[0] == 4: h_averageProtoL4_S.SetBinContent(cell[1],cell[2],1)
     if cell[0] == 5: h_averageProtoL5_S.SetBinContent(cell[1],cell[2],1)
 
    for cell in NN:
     if cell[0] == 0: h_averageProtoL0_N.SetBinContent(cell[1],cell[2],20)
     if cell[0] == 1: h_averageProtoL1_N.SetBinContent(cell[1],cell[2],20)
     if cell[0] == 2: h_averageProtoL2_N.SetBinContent(cell[1],cell[2],20)
     if cell[0] == 3: h_averageProtoL3_N.SetBinContent(cell[1],cell[2],20)
     if cell[0] == 4: h_averageProtoL4_N.SetBinContent(cell[1],cell[2],20)
     if cell[0] == 5: h_averageProtoL5_N.SetBinContent(cell[1],cell[2],20)


    for cell in PP:
     if cell[0] == 0: h_averageProtoL0_P.SetBinContent(cell[1],cell[2],30)
     if cell[0] == 1: h_averageProtoL1_P.SetBinContent(cell[1],cell[2],30)
     if cell[0] == 2: h_averageProtoL2_P.SetBinContent(cell[1],cell[2],30)
     if cell[0] == 3: h_averageProtoL3_P.SetBinContent(cell[1],cell[2],30)
     if cell[0] == 4: h_averageProtoL4_P.SetBinContent(cell[1],cell[2],30)
     if cell[0] == 5: h_averageProtoL5_P.SetBinContent(cell[1],cell[2],30)


