#import ROOT
from sys import exit
import random
from array import array
import numpy as np
from operator import itemgetter
#import numpy as np

####################

def Clustering(layer1,layer2,layer3,layer4,layer5,layer6):
 SeedS = Convert(layer1,layer2,layer3,layer4,layer5,layer6,5,100000000000000)
 SeedN = Convert(layer1,layer2,layer3,layer4,layer5,layer6,2,5)
 SeedP = Convert(layer1,layer2,layer3,layer4,layer5,layer6,0,2)

 #print("printing seedS",SeedS)  

 PC = Proto(SeedS,SeedN,SeedP)
 print('Proto before merge',len(PC), PC)
 print('SeedS', SeedS)
 if len(PC)==1: PCm = PC 
 else: PCm = MergeS(PC,SeedS)

 #print("PC after merge",len(PC), PCm)
 return PCm


####################


def Convert(layer1,layer2,layer3,layer4,layer5,layer6,lower,upper):
 seed = []

 #!!!!
 #noise threshold for Noisless sample
 #noise=[1, 1, 1, 1, 1, 1]

 #noise threshold for Noisy samples
 noise = [13, 34, 17, 14,  8, 14]

 
# print(layer1[0])
 layer = []
 layer.append(layer1.tolist())
 layer.append(layer2.tolist())
 layer.append(layer3.tolist())
 layer.append(layer4.tolist())
 layer.append(layer5.tolist())
 layer.append(layer6.tolist())

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
    #if cellX>0print("Seed E", celE[3],"Greater than ",noise[lt]*lower , Y, X)
    if celE[3]>noise[lt]* lower and celE[3]<noise[lt] * upper:
       seed.append([lt,Y,X])

 #print("printing seed",seed) 
 return seed


#########################



def Proto(sedS,sedN,sedP):
    proto_ensemble = []

    #check 2 neighbor if ATL==1 else check 1 neighbor

    ATLS=True

    ATLP=False
    for cell in sedS:
     proto = []
     cell_proto = []
     lay = cell[0]
     cX  = cell[1]
     cY  = cell[2]
     proto.append(cell)
     #define the one close to it
     #need to add new ones

     layers=[64,32,32,16,16,8]

     #print("CELL", cell)
     if lay!=5:
      nextl(lay,cell_proto,"next",cX,cY,ATLS)
     if lay!=0:
      nextl(lay,cell_proto,"prev",cX,cY,ATLS)

     nextl(lay,cell_proto,"same",cX,cY,ATLS)
  
     #print("NB",proto)
     
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

       if layNP!=5: nextl(layNP,new_CellP,"next",cXNP,cYNP,ATLP)
       if layNP!=0: nextl(layNP,new_CellP,"prev",cXNP,cYNP,ATLP)
       nextl(layNP,new_CellP,"same",cXNP,cYNP,ATLP)
       
       
       for new_cell in new_CellP:
        if new_cell not in proto:
         if new_cell in sedP or new_cell in sedN or new_cell in sedS:
                  
          proto.append(new_cell)

     
     #print('protoensamble',proto)
     proto_ensemble.append(proto)
    #need now to merge
 
    for i in range(50):
     proto_ensemble=SMERGE(proto_ensemble)
     proto_ensemble=clean_duplicates(proto_ensemble)

  

    return proto_ensemble


########################

def MergeS(proto,seed):

 lung1=len(proto)
 lung2=-1
 #print(len(proto))
 for i in range(50):
  proto=Iter_2(proto,seed)
  proto=clean_duplicates(proto)

 return proto

########################

def nextl(l,prot,which, X, Y, ATL):
    
    if which=="same":
        prot.append([l,X+1 ,Y+1])
        prot.append([l,X   ,Y+1])
        prot.append([l,X-1 ,Y+1])
        prot.append([l,X-1 ,Y  ])
        prot.append([l,X-1 ,Y-1])
        prot.append([l,X   ,Y-1])
        prot.append([l,X+1 ,Y-1])
        prot.append([l,X+1 ,Y  ])
        
        if ATL==True:
            prot.append([l,X+2 ,Y  ])
            prot.append([l,X+2 ,Y+1])
            prot.append([l,X+1 ,Y+2])
            prot.append([l,X   ,Y+2])
            prot.append([l,X-1 ,Y+2])
            prot.append([l,X-2 ,Y+1])
            prot.append([l,X-2 ,Y  ])
            prot.append([l,X-2 ,Y-1])
            prot.append([l,X-1 ,Y-2])
            prot.append([l,X   ,Y-2])
            prot.append([l,X+1 ,Y-2])
            prot.append([l,X+2 ,Y-1])
        return
       
    layers=[64,32,32,16,16,8]
    if which=="next": 
     rat=layers[l+1]/float(layers[l])
     l=l+1
   
    elif which=="prev":
 
        rat=layers[l-1]/float(layers[l])
        l=l-1

    if rat==2:
        prot.append([l,2*X+1 ,2*Y+1])
        prot.append([l,2*X ,  2*Y+1])
        prot.append([l,2*X+1 ,2*Y])
        prot.append([l,2*X ,  2*Y])
    if rat==1:

        prot.append([l,X ,  Y])

        
    if rat==0.5:
        prot.append([l,int(X/2//1),int(Y/2//1)])

#######################

def common_cluster(proto1,proto2):
    protoO = proto1
    for cel in proto1:
     if cel in proto2:

       protoO = proto1+proto2
       break
    return protoO,proto1,proto2

   
#######################

def clean_duplicates(merge_final):
   cleaned_m = []
   for m in merge_final:
     cleaned = []
     for c in m:
      if c not in cleaned:
       cleaned.append(c)
     cleaned_m.append(cleaned)
   return cleaned_m

#######################

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


#####################


def Iter_2(my_ens,sed):
 new_ens=[]
 #check=[]
 check=[0]*len(my_ens)
 for i,p in enumerate(my_ens):
  clus=p
  for l,pp in enumerate(my_ens):
   if l>i:

    cell1=[]
    for c in p:
     #print([c[0]+1,2*c[1]+1,2*c[2]+1] not in p and [c[0]+1,2*c[1]+1,2*c[2]+1] not in cell1)                                                                                                               

     if c[0]==5 or c[0] ==1 or c[0] ==3:
      if [c[0]-1,2*c[1]+1,2*c[2]+1] not in p and [c[0]-1,2*c[1]+1,2*c[2]+1] not in cell1: cell1.append([c[0]-1,2*c[1]+1,2*c[2]+1])
      if [c[0]-1,2*c[1],2*c[2]+1]   not in p and [c[0]-1,2*c[1],2*c[2]+1]   not in cell1: cell1.append([c[0]-1,2*c[1],2*c[2]+1])
      if [c[0]-1,2*c[1]+1,2*c[2]]   not in p and [c[0]-1,2*c[1]+1,2*c[2]]   not in cell1: cell1.append([c[0]-1,2*c[1]+1,2*c[2]])
      if [c[0]-1,2*c[1],2*c[2]]     not in p and [c[0]-1,2*c[1],2*c[2]]     not in cell1: cell1.append([c[0]-1,2*c[1],2*c[2]])

     if c[0]==0  or c[0] ==2 or c[0] ==4:
      if [c[0]+1,int(c[1]/2//1),int(c[2]/2//1)] not in p and [c[0]+1,int(c[1]/2//1),int(c[2]/2//1)] not in cell1: cell1.append([c[0]+1,int(c[1]/2//1),int(c[2]/2//1)] )

     if c[0] == 3 or c[0]==1:
      if [c[0]+1,c[1],c[2]] not in p and [c[0]+1,c[1],c[2]] not in cell1: cell1.append([c[0]+1,c[1],c[2]])

     if c[0] == 4 or c[0]==2:
      if [c[0]-1,c[1],c[2]] not in p and [c[0]-1,c[1],c[2]] not in cell1: cell1.append([c[0]-1,c[1],c[2]])

     if [c[0],c[1]+1,c[2]] not in p and [c[0],c[1]+1,c[2]] not in cell1: cell1.append([c[0],c[1]+1,c[2]])
     if [c[0],c[1]-1,c[2]] not in p and [c[0],c[1]-1,c[2]] not in cell1: cell1.append([c[0],c[1]-1,c[2]])
     if [c[0],c[1],c[2]+1] not in p and [c[0],c[1],c[2]+1] not in cell1: cell1.append([c[0],c[1],c[2]+1])
     if [c[0],c[1],c[2]-1] not in p and [c[0],c[1],c[2]-1] not in cell1: cell1.append([c[0],c[1],c[2]-1])

    cell2=[]
    for c in pp:
     
     if c[0]==5 or c[0] ==1 or c[0] ==3:
      if [c[0]-1,2*c[1]+1,2*c[2]+1] not in pp and [c[0]-1,2*c[1]+1,2*c[2]+1] not in cell2: cell2.append( [c[0]-1,2*c[1]+1,2*c[2]+1])
      if [c[0]-1,2*c[1],2*c[2]+1]   not in pp and [c[0]-1,2*c[1],2*c[2]+1]   not in cell2: cell2.append( [c[0]-1,2*c[1],2*c[2]+1])
      if [c[0]-1,2*c[1]+1,2*c[2]]   not in pp and [c[0]-1,2*c[1]+1,2*c[2]]   not in cell2: cell2.append( [c[0]-1,2*c[1]+1,2*c[2]])
      if [c[0]-1,2*c[1],2*c[2]]     not in pp and [c[0]-1,2*c[1],2*c[2]]     not in cell2: cell2.append( [c[0]-1,2*c[1],2*c[2]])
     if c[0]==0 or c[0] ==2 or c[0] ==4:
      if [c[0]+1,int(c[1]/2//1),int(c[2]/2//1)] not in pp and [c[0]+1,int(c[1]/2//1),int(c[2]/2//1)] not in cell2: cell2.append( [c[0]+1,int(c[1]/2//1),int(c[2]/2//1)] )
     if c[0] == 3 or c[0]==1:
      if [c[0]+1,c[1],c[2]] not in pp and [c[0]+1,c[1],c[2]] not in cell2: cell2.append( [c[0]+1,c[1],c[2]])
     if c[0] == 4 or c[0]==2:
      if [c[0]-1,c[1],c[2]] not in pp and [c[0]-1,c[1],c[2]] not in cell2: cell2.append( [c[0]-1,c[1],c[2]]   )

     if [c[0],c[1]+1,c[2]] not in pp and [c[0],c[1]+1,c[2]] not in cell2: cell2.append([c[0],c[1]+1,c[2]])
     if [c[0],c[1]-1,c[2]] not in pp and [c[0],c[1]-1,c[2]] not in cell2: cell2.append([c[0],c[1]-1,c[2]])
     if [c[0],c[1],c[2]+1] not in pp and [c[0],c[1],c[2]+1] not in cell2: cell2.append([c[0],c[1],c[2]+1])
     if [c[0],c[1],c[2]-1] not in pp and [c[0],c[1],c[2]-1] not in cell2: cell2.append([c[0],c[1],c[2]-1])

    for neibp in cell1:
     if (neibp in pp and neibp in sed):
      clus+=pp
      check[l]+=1
      break
    for neibpp in cell2:
     if(neibpp in p and neibpp in sed):
      clus+=pp
      check[l]+=1
      break
   if my_ens[l]==my_ens[-1] and check[i]==0:
    new_ens.append(clus)

 return new_ens


###############################


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

#############################

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


#################################
