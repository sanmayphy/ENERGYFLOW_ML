import ROOT
from sys import exit
import random
from array import array
import numpy as np
from operator import itemgetter
#import numpy as np
def Clustering(layer1,layer2,layer3,layer4,layer5,layer6):
 SeedS = Convert(layer1,layer2,layer3,layer4,layer5,layer6,5,100000000000000)
 SeedN = Convert(layer1,layer2,layer3,layer4,layer5,layer6,2,5)
 SeedP = Convert(layer1,layer2,layer3,layer4,layer5,layer6,0,2)

 PC = Proto(SeedS,SeedN,SeedP)
 PCm = MergeS(PC,SeedS)
 
 return PCm



def Convert(layer1,layer2,layer3,layer4,layer5,layer6,lower,upper):
 seed = []
 noise = [10, 20, 15, 20, 20, 20]

 layer = []
 layer.append(list(layer1[0]))
 layer.append(list(layer2[0]))
 layer.append(list(layer3[0]))
 layer.append(list(layer4[0]))
 layer.append(list(layer5[0]))
 layer.append(list(layer6[0]))


 lt = - 1 
 for l in layer:
  lt = lt +1
  Y = -1
  for cellY in l:
   Y = Y +1
   X = -1
   for cellX in cellY:
    X = X + 1
    celE = [lt,Y,X,cellX]
    if celE[3]/noise[lt] > lower and celE[3]/noise[lt] < upper:
       seed.append([lt,Y,X])
  
 return seed






def Proto(sedS,sedN,sedP):
    proto_ensemble = []
    for cell in sedS:
     proto = []
     cell_proto = []
     lay = cell[0]
     cX  = cell[1]
     cY  = cell[2]
     proto.append(cell)
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


def Assign_Topo(Proto):
    #out_image1 = np.zeros( [32, 32] )
    #out_image2 = np.zeros( [64, 64] )
    #out_image3 = np.zeros( [32, 32] )
    #out_image4 = np.zeros( [16, 16] )
    #out_image5 = np.zeros( [16, 16] )
    #out_image6 = np.zeros( [8, 8] )
    out_image1 = np.zeros( [64,64] )
    out_image2 = np.zeros( [64,64] )
    out_image3 = np.zeros( [64,64] )
    out_image4 = np.zeros( [64,64] )
    out_image5 = np.zeros( [64,64] )
    out_image6 = np.zeros( [64,64] )


    np1 = 0
    for p in Proto: 
     np1 = np1 + 1
     for c in p:
      if c[0] == 0:
       out_image1[c[1]-1,c[2]-1] = np1    
      if c[0] == 1:
       out_image2[c[1]-1,c[2]-1] = np1      
      if c[0] == 2:
       out_image3[c[1]-1,c[2]-1] = np1      
      if c[0] == 3:
       out_image4[c[1]-1,c[2]-1] = np1      
      if c[0] == 4:
       out_image5[c[1]-1,c[2]-1] = np1      
      if c[0] == 5:
       out_image6[c[1]-1,c[2]-1] = np1      
       

    out_imagef = [out_image1,out_image2,out_image3,out_image4,out_image5,out_image6]

    return np.stack(out_imagef,axis=0)

