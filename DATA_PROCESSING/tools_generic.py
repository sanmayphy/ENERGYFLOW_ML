#import ROOT
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

 #print("printing seedS",SeedS)  
 #print("printing seedN",SeedN)
 #print("printing seedP",SeedP)
 PC = Proto(SeedS,SeedN,SeedP)
 #print('PC before merge',len(PC), PC)
 if len(PC)==1: PCm = PC 
 else: PCm = MergeS(PC,SeedS)

 #print("PC after merge",len(PC), PCm)
# print("PCC",PCm)
 return PCm



def Convert(layer1,layer2,layer3,layer4,layer5,layer6,lower,upper):
 seed = []
 noise = [20, 20, 30, 80, 80, 160]

# print(layer1[0])
 layer = []
 layer.append(layer1[0].tolist())
 layer.append(layer2[0].tolist())
 layer.append(layer3[0].tolist())
 layer.append(layer4[0].tolist())
 layer.append(layer5[0].tolist())
 layer.append(layer6[0].tolist())

 #print('layer 5',layer[4])
 #print("En. Check")
 lt = - 1 
 for l in layer:
  lt = lt +1
  
#  if lt == 0: print('printing layer', lt, l) 
  Y = -1
  for cellY in l:
   Y = Y +1
   X = -1
   for cellX in cellY:
    X = X + 1
    celE = [lt,Y,X,cellX]
#    if lt==1 and celE[3]/noise[lt]>3: print("Seed E", celE[3],"Greater than ",noise[lt]*lower , Y, X)
    if celE[3]/noise[lt] > lower and celE[3]/noise[lt] < upper:
       seed.append([lt,Y,X])

 #print("printing seed",seed) 
 return seed






def Proto(sedS,sedN,sedP):
    proto_ensemble = []

    #check 2 neighbor if ATL==1 else check 1 neighbor
    ATL = 1
    
    for cell in sedS:
     proto = []
     cell_proto = []
     lay = cell[0]
     cX  = cell[1]
     cY  = cell[2]
     proto.append(cell)
     #define the one close to it
     #need to add new ones

     
     #+1 layer doubled
     if lay == 0:
      cell_proto.append([lay+1,2*cX+1,2*cY+1])
      cell_proto.append([lay+1,2*cX+1,2*cY])
      cell_proto.append([lay+1,2*cX,2*cY+1])
      cell_proto.append([lay+1,2*cX,2*cY])

     if lay == 3:
      cell_proto.append([lay+1,cX,cY])

     if lay == 4:
       cell_proto.append([lay-1,cX,cY])
     #-1 layer doubled 
     if lay==5 or lay ==2 or lay ==3:
      cell_proto.append([lay-1,2*cX+1,2*cY+1])
      cell_proto.append([lay-1,2*cX+1,2*cY])
      cell_proto.append([lay-1,2*cX,2*cY+1])
      cell_proto.append([lay-1,2*cX,2*cY])

     #+1 layer half 
     if lay==1 or lay ==2  or lay ==4:
      cell_proto.append([lay+1,int(cX/2//1),int(cY/2//1)])

     #-1 layer half
     if lay == 1:
      cell_proto.append([lay-1,int(cX/2//1),int(cY/2//1)])

     cell_proto.append([lay,cX+1,cY])
     cell_proto.append([lay,cX-1,cY])
     cell_proto.append([lay,cX,cY+1])
     cell_proto.append([lay,cX,cY-1])
     cell_proto.append([lay,cX+1,cY+1])
     cell_proto.append([lay,cX+1,cY-1])
     cell_proto.append([lay,cX-1,cY+1])
     cell_proto.append([lay,cX-1,cY-1])

     if ATL == 1:
      cell_proto.append([lay,cX+2,cY])
      cell_proto.append([lay,cX-2,cY])
      cell_proto.append([lay,cX,cY+2])
      cell_proto.append([lay,cX,cY-2])
      cell_proto.append([lay,cX+1,cY+2])
      cell_proto.append([lay,cX+2,cY+1])
      cell_proto.append([lay,cX+1,cY-2])
      cell_proto.append([lay,cX+2,cY-1])
      cell_proto.append([lay,cX-1,cY+2])
      cell_proto.append([lay,cX-2,cY+1])
      cell_proto.append([lay,cX-1,cY-2])
      cell_proto.append([lay,cX-2,cY-1])

     
     #first cluster the low energy ones
     for cp in cell_proto:
      if cp  in sedP:
       #print "increasing the protoCluster with seed",cell," and new cell P:",cp
       proto.append(cp)
     for cpN in cell_proto:
      if cpN in sedN or cpN in sedS:
       
       #print "increasing the protoCluster with seed",cell," and new cell N:",cp
       proto.append(cpN)
   
       layNP = cpN[0]
       cXNP  = cpN[1]
       cYNP  = cpN[2]
       new_CellP = []
       if layNP == 0:
        new_CellP.append([layNP+1,2*cXNP+1,2*cYNP+1])
        new_CellP.append([layNP+1,2*cXNP+1,2*cYNP])
        new_CellP.append([layNP+1,2*cXNP,2*cYNP+1])
        new_CellP.append([layNP+1,2*cXNP,2*cYNP])

       if layNP==2:
        new_CellP.append([layNP-1,2*cXNP+1,2*cYNP+1])
	new_CellP.append([layNP-1,2*cXNP+1,2*cYNP])
        new_CellP.append([layNP-1,2*cXNP,2*cYNP+1])
        new_CellP.append([layNP-1,2*cXNP,2*cYNP])
        
        new_CellP.append([layNP+1,cXNP/2//1,cYNP/2//1])

       if layNP==1:
        new_CellP.append([layNP+1,int(cXNP/2//1),int(cYNP/2//1)])
        new_CellP.append([layNP-1,int(cXNP/2//1),int(cYNP/2//1)])

       if layNP ==3:
        new_CellP.append([layNP-1,2*cXNP+1,2*cYNP+1])
        new_CellP.append([layNP-1,2*cXNP+1,2*cYNP])
        new_CellP.append([layNP-1,2*cXNP,2*cYNP+1])
        new_CellP.append([layNP-1,2*cXNP,2*cYNP])

        new_CellP.append([layNP+1,cXNP,cYNP]) 

       if layNP ==4:
        new_CellP.append([layNP-1,cXNP,cYNP])

        new_CellP.append([layNP+1,int(cXNP/2//1),int(cYNP/2//1)])

       if lay ==5:
        new_CellP.append([layNP-1,2*cXNP+1,2*cYNP+1])
	new_CellP.append([layNP-1,2*cXNP+1,2*cYNP])
        new_CellP.append([layNP-1,2*cXNP,2*cYNP+1])
        new_CellP.append([layNP-1,2*cXNP,2*cYNP])
        

       new_CellP.append([layNP,cXNP+1,cYNP])
       new_CellP.append([layNP,cXNP-1,cYNP])
       new_CellP.append([layNP,cXNP,cYNP+1])
       new_CellP.append([layNP,cXNP,cYNP-1])
       new_CellP.append([layNP,cXNP+1,cYNP+1])
       new_CellP.append([layNP,cXNP+1,cYNP-1])
       new_CellP.append([layNP,cXNP-1,cYNP+1])
       new_CellP.append([layNP,cXNP-1,cYNP-1])

       #if ATL==1:
        #new_CellP.append([layNP,cXNP+2,cYNP])                                                                                                                                                             
        #new_CellP.append([layNP,cXNP-2,cYNP])                                                                                                                                                             
        #new_CellP.append([layNP,cXNP,cYNP+2])                                                                                                                                                              
        #new_CellP.append([layNP,cXNP,cYNP-2]) 
        #new_CellP.append([layNP,cXNP+1,cYNP+2])
        #new_CellP.append([layNP,cXNP+2,cYNP+1])
        #new_CellP.append([layNP,cXNP+1,cYNP-2])
        #new_CellP.append([layNP,cXNP-2,cYNP+1])
        #new_CellP.append([layNP,cXNP-1,cYNP+2])
        #new_CellP.append([layNP,cXNP+2,cYNP-1])
        #new_CellP.append([layNP,cXNP-1,cYNP-2])
        #new_CellP.append([layNP,cXNP-2,cYNP-1])
       
       for new_cell in new_CellP:
        if new_cell not in proto:
         if new_cell in sedP or new_cell in sedN or new_cell in sedS:
                  
          proto.append(new_cell)

     
     #print('protoensamble',proto)
     proto_ensemble.append(proto)
    #need now to merge
     

    #print("PROTOENSEMBLE B",proto_ensemble)

    #l1=len(proto_ensemble)
    #l2=-1
    for i in range(50):
     #l1=len(proto_ensemble)
     proto_ensemble=SMERGE(proto_ensemble)
     proto_ensemble=clean_duplicates(proto_ensemble)
     #l2=len(proto_ensemble)

    #print("Proto A",proto_ensemble, len(proto_ensemble))
     
    #merge_final = []
    #merge_final_check = []
   
    #i = 0
    #f = 0
    #for pe in proto_ensemble:
    # i = i +1
    # merged = pe
    # j = 0
    # if pe not in merge_final_check:
    #  for pe_1 in proto_ensemble:
    #   j = j +1
    #   if pe_1 != pe and pe_1 not in merge_final_check:
    #    merged,m_app1,m_app2 = common_cluster(merged,pe_1)
    #    if merged != m_app1:
    #    merge_final_check.append(pe)
    #     merge_final_check.append(pe_1)
    #  if merged != []: f = f + 1
    #  if merged != []: merge_final.append(merged)

#        merged = merged + common_cluster(pe,pe_1)
#      proto_ensembleMerged.append(merged)
#
    #merge_clean = clean_duplicates(proto_ensamble)
    #print("CLEANN",merge_clean)


    return proto_ensemble

def MergeS(proto,seed):

 lung1=len(proto)
 lung2=-1
 #print(len(proto))
 for i in range(1000):
  lung1=len(proto)
  Proto=Iter(proto,seed)
  proto=clean_duplicates(proto)
  lung2=len(proto)
  if lung1==lung2: break

 return proto






# p_check = []
# p_f = []
# appo = []
# for p in proto:
#  pn = p
#  p_check.append(p)
 # for p1 in proto:
 #  if p1 != p and p1 not in p_check:
 #   cell1= []
 #   for c in p1:
 #    if c[0] == 0:
 #     cell1.append([c[0]+1,2*c[1]+1,2*c[2]+1])
 #     cell1.append([c[0]+1,2*c[1]+1,2*c[2]])
 #     cell1.append([c[0]+1,2*c[1],2*c[2]+1])
 #     cell1.append([c[0]+1,2*c[1],2*c[2]])
      
 #    if c[0]==5 or c[0] ==2 or c[0] ==3:
 #     cell1.append([c[0]-1,2*c[1]+1,2*c[2]+1])
 #     cell1.append([c[0]-1,2*c[1]+1,2*c[2]])
 #     cell1.append([c[0]-1,2*c[1],2*c[2]+1])
 #     cell1.append([c[0]-1,2*c[1],2*c[2]])
      
#     if c[0]==1 or c[0] ==2 or c[0] ==4:
      #cell1.append([c[0]+1,c[1]/2//1,c[2]/2//1]) 
        
     #if c[0] == 1:
      #cell1.append([c[0]-1,c[1]/2//1,c[2]/2//1])
      
  #   if c[0] == 3:
  #    cell1.append([c[0]+1,c[1],c[2]])
        
  #   if c[0] == 4:
  #    cell1.append([c[0]-1,c[1],c[2]])
      
   #  cell1.append([c[0],c[1]+1,c[2]])
  #   cell1.append([c[0],c[1]-1,c[2]])
  #   cell1.append([c[0],c[1],c[2]+1])
  #   cell1.append([c[0],c[1],c[2]-1])

#    print("BOUNDARY",cell1)
     
   # for neib in cell1:
   #  if neib in p and neib in seed:
     # pn=pn+p1
     #else
      #pf.append(p1)

    #pf.append(pn)

      

      
  #    p_f.append(p + p1)
  #    p_check.append(p)
  #    p_check.append(p1)
  #    break
# if p_f ==[]: p_f = proto
 
# return p_f
     


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
 #    print(np1)
     for c in p:
      if c[0] == 0:
       out_image1[c[1],c[2]] = np1    
      if c[0] == 1:
       out_image2[c[1],c[2]] = np1      
      if c[0] == 2:
       out_image3[c[1],c[2]] = np1      
      if c[0] == 3:
       out_image4[c[1],c[2]] = np1      
      if c[0] == 4:
       out_image5[c[1],c[2]] = np1      
      if c[0] == 5:
       out_image6[c[1],c[2]] = np1      

    
    out_imagef = [out_image1,out_image2,out_image3,out_image4,out_image5,out_image6]
#    if(np1>1):
#     print(out_imagef)
    return out_imagef
    #return np.stack(out_imagef,axis=0)





def Iter(my_ens,sed):
 new_ens=[]
 check=[]
 k=0
 for i,p in enumerate(my_ens):
  check.append(p)
  j=0  
  for l,pp in enumerate(my_ens):
   if p!=pp and pp not in check:
    
    cell1=[]
    for c in p:
     #print([c[0]+1,2*c[1]+1,2*c[2]+1] not in p and [c[0]+1,2*c[1]+1,2*c[2]+1] not in cell1)
     if c[0] == 0:
      if [c[0]+1,2*c[1]+1,2*c[2]+1] not in p and [c[0]+1,2*c[1]+1,2*c[2]+1] not in cell1: cell1.append([c[0]+1,2*c[1]+1,2*c[2]+1])
      if [c[0]+1,2*c[1],2*c[2]+1]   not in p and [c[0]+1,2*c[1],2*c[2]+1]   not in cell1: cell1.append([c[0]+1,2*c[1],2*c[2]+1])
      if [c[0]+1,2*c[1]+1,2*c[2]]   not in p and [c[0]+1,2*c[1]+1,2*c[2]]   not in cell1: cell1.append([c[0]+1,2*c[1]+1,2*c[2]])
      if [c[0]+1,2*c[1],2*c[2]]     not in p and [c[0]+1,2*c[1],2*c[2]]     not in cell1: cell1.append([c[0]+1,2*c[1],2*c[2]] )
     if c[0]==5 or c[0] ==2 or c[0] ==3:
      if [c[0]-1,2*c[1]+1,2*c[2]+1] not in p and [c[0]-1,2*c[1]+1,2*c[2]+1] not in cell1: cell1.append([c[0]-1,2*c[1]+1,2*c[2]+1])
      if [c[0]-1,2*c[1],2*c[2]+1]   not in p and [c[0]-1,2*c[1],2*c[2]+1]   not in cell1: cell1.append([c[0]-1,2*c[1],2*c[2]+1])
      if [c[0]-1,2*c[1]+1,2*c[2]]   not in p and [c[0]-1,2*c[1]+1,2*c[2]]   not in cell1: cell1.append([c[0]-1,2*c[1]+1,2*c[2]])
      if [c[0]-1,2*c[1],2*c[2]]     not in p and [c[0]-1,2*c[1],2*c[2]]     not in cell1: cell1.append([c[0]-1,2*c[1],2*c[2]])
     if c[0]==1 or c[0] ==2 or c[0] ==4:
      if [c[0]+1,int(c[1]/2//1),int(c[2]/2//1)] not in p and [c[0]+1,int(c[1]/2//1),int(c[2]/2//1)] not in cell1: cell1.append([c[0]+1,int(c[1]/2//1),int(c[2]/2//1)] )
     if c[0] == 1:
      if [c[0]-1,int(c[1]/2//1),int(c[2]/2//1)] not in p and [c[0]-1,int(c[1]/2//1),int(c[2]/2//1)] not in cell1: cell1.append([c[0]-1,int(c[1]/2//1),int(c[2]/2//1)] )
     if c[0] == 3:
      if [c[0]+1,c[1],c[2]] not in p and [c[0]+1,c[1],c[2]] not in cell1: cell1.append([c[0]+1,c[1],c[2]])
     if c[0] == 4:
      if [c[0]-1,c[1],c[2]] not in p and [c[0]-1,c[1],c[2]] not in cell1: cell1.append([c[0]-1,c[1],c[2]])
      if [c[0],c[1]+1,c[2]] not in p and [c[0],c[1]+1,c[2]] not in cell1: cell1.append([c[0],c[1]+1,c[2]])
      if [c[0],c[1]-1,c[2]] not in p and [c[0],c[1]-1,c[2]] not in cell1: cell1.append([c[0],c[1]-1,c[2]])
      if [c[0],c[1],c[2]+1] not in p and [c[0],c[1],c[2]+1] not in cell1: cell1.append([c[0],c[1],c[2]+1])
      if [c[0],c[1],c[2]-1] not in p and [c[0],c[1],c[2]-1] not in cell1: cell1.append([c[0],c[1],c[2]-1])
                
    cell2=[]
    for c in pp:
     if c[0] == 0:
      if [c[0]+1,2*c[1]+1,2*c[2]+1] not in pp and [c[0]+1,2*c[1]+1,2*c[2]+1] not in cell2: cell2.append([c[0]+1,2*c[1]+1,2*c[2]+1])
      if [c[0]+1,2*c[1],2*c[2]+1]   not in pp and [c[0]+1,2*c[1],2*c[2]+1]   not in cell2: cell2.append( [c[0]+1,2*c[1],2*c[2]+1])
      if [c[0]+1,2*c[1]+1,2*c[2]]   not in pp and [c[0]+1,2*c[1]+1,2*c[2]]   not in cell2: cell2.append( [c[0]+1,2*c[1]+1,2*c[2]])
      if [c[0]+1,2*c[1],2*c[2]]     not in pp and [c[0]+1,2*c[1],2*c[2]]     not in cell2: cell2.append( [c[0]+1,2*c[1],2*c[2]]  )
     if c[0]==5 or c[0] ==2 or c[0] ==3:
      if [c[0]-1,2*c[1]+1,2*c[2]+1] not in pp and [c[0]-1,2*c[1]+1,2*c[2]+1] not in cell2: cell2.append( [c[0]-1,2*c[1]+1,2*c[2]+1])
      if [c[0]-1,2*c[1],2*c[2]+1]   not in pp and [c[0]-1,2*c[1],2*c[2]+1]   not in cell2: cell2.append( [c[0]-1,2*c[1],2*c[2]+1])
      if [c[0]-1,2*c[1]+1,2*c[2]]   not in pp and [c[0]-1,2*c[1]+1,2*c[2]]   not in cell2: cell2.append( [c[0]-1,2*c[1]+1,2*c[2]])
      if [c[0]-1,2*c[1],2*c[2]]     not in pp and [c[0]-1,2*c[1],2*c[2]]     not in cell2: cell2.append( [c[0]-1,2*c[1],2*c[2]])
     if c[0]==1 or c[0] ==2 or c[0] ==4:
      if [c[0]+1,int(c[1]/2//1),int(c[2]/2//1)] not in pp and [c[0]+1,int(c[1]/2//1),int(c[2]/2//1)] not in cell2: cell2.append( [c[0]+1,int(c[1]/2//1),int(c[2]/2//1)] )
     if c[0] == 1:
      if [c[0]-1,int(c[1]/2//1),int(c[2]/2//1)] not in pp and [c[0]-1,int(c[1]/2//1),int(c[2]/2//1)] not in cell2: cell2.append([c[0]-1,int(c[1]/2//1),int(c[2]/2//1)] )
     if c[0] == 3:
      if [c[0]+1,c[1],c[2]] not in pp and [c[0]+1,c[1],c[2]] not in cell2: cell2.append( [c[0]+1,c[1],c[2]])
     if c[0] == 4:
      if [c[0]-1,c[1],c[2]] not in pp and [c[0]-1,c[1],c[2]] not in cell2: cell2.append( [c[0]-1,c[1],c[2]]   )
      if [c[0],c[1]+1,c[2]] not in pp and [c[0],c[1]+1,c[2]] not in cell2: cell2.append([c[0],c[1]+1,c[2]])
      if [c[0],c[1]-1,c[2]] not in pp and [c[0],c[1]-1,c[2]] not in cell2: cell2.append([c[0],c[1]-1,c[2]])
      if [c[0],c[1],c[2]+1] not in pp and [c[0],c[1],c[2]+1] not in cell2: cell2.append([c[0],c[1],c[2]+1])
      if [c[0],c[1],c[2]-1] not in pp and [c[0],c[1],c[2]-1] not in cell2: cell2.append([c[0],c[1],c[2]-1])
      
      
    for neibp in cell1:
     if (neibp in pp and neibp in sed):
      new_ens.append(p+pp)
      check.append(pp)
      j=1
      k+=1
      break
    for neibpp in cell2:
     if(neibpp in p and neibpp in sed and j==0):
      new_ens.append(p+pp)
      check.append(pp)
      k+=1
      break
     elif j==0: 
      new_ens.append(pp)
      check.append(pp)
      break  
  if k==0 and p not in new_ens:
   new_ens.append(p)
    
 return new_ens










def StepMerge(my_ens):
    new_ens=[]
    check=[]
    k=0
    for i,p in enumerate(my_ens):
        check.append(p)
        j=0  
        for l,pp in enumerate(my_ens):
            if p!=pp and pp not in check:
                for neibpp in pp:
                    
                    #print(neibpp, l, (neibpp in p))
                    if(neibpp in p ):
                        #print(neibpp, l)
                        new_ens.append(p+pp)
                        check.append(pp)
                        k+=1
                        j+=1
                        break
                if j==0: 
                    new_ens.append(pp)
                    check.append(pp)
  
        if k==0 and p not in new_ens:
            new_ens.append(p)
    
    return new_ens



def SMERGE(my_ens):
    new_ens=[]
    check=[0]*len(my_ens)
    for i,p in enumerate(my_ens):
        clus=p
        for l,pp in enumerate(my_ens):
            if l>i:
                for neibpp in pp:
                    if(neibpp in p) :
                        clus+=pp
                        check[l]+=1
                        break
            if my_ens[l]==my_ens[-1] and check[i]==0:
                new_ens.append(clus)
    
    return new_ens
   

def MergeStep(my_ens):
 new_ens=[]
 check=[]
 k=0
 for i,p in enumerate(my_ens):
  check.append(p)
  j=0  
  for l,pp in enumerate(my_ens):
   if p!=pp and pp not in check:
      
 
    for intpp in pp:
     if(intpp in p  and j==0):
      new_ens.append(p+pp)
      check.append(pp)
      #print("merged"+str(i)+str(l),intpp)
      k+=1
      j+=1
      break
    if j==0: 
     new_ens.append(pp)
     check.append(pp)
     
  if k==0 and p not in new_ens:
   new_ens.append(p)
    
 return new_ens
