#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:37:25 2019

@author: yuanneu
"""

from NodeKeras import *


timefa1=float(sys.argv[1])
seed_num=int(timefa1)
timefa2=float(sys.argv[2])
epoc_num=int(timefa2)
timefa3=float(sys.argv[3])
lam=1*timefa3
timefa4=float(sys.argv[3])
dim=int(timefa4)




random.seed(seed_num)
iter_num=5


A=np.zeros((52,52))
for i in range(26):
    for j in range(i+1,26):
        z=np.random.randint(2)
        if z==1:
            A[i,j]=1
            A[j,i]=1
        
for i in range(26,39):
    for j in range(39,52):
        A[i,j]=1
        A[j,i]=1
        
for i in range(26):
    A[i,i+26]=1
    A[i+26,i]=1
    


ablist=[i for i in range(52)]  
random.shuffle(ablist)
P=np.zeros((52,52))
for i in range(52):
    P[i,ablist[i]]=1
#P=np.identity(36)
B=np.dot(np.dot(P.T,A),P)
ddn=0
while (ddn<=14):
    i,j=random.sample(ablist,2)
    if (B[i,j]==1):
        B[i,j]=0
        B[j,i]=0
        ddn+=1
    else:
        pass

ddn=0
while (ddn<=14):
    i,j=random.sample(ablist,2)
    if (A[i,j]==1):
        A[i,j]=0
        A[j,i]=0
        ddn+=1
    else:
        pass


ya=np.zeros((52))
for i in range(2):
    for j in range(26):
        ya[26*i+j]=i
yb=np.dot(P.T,ya)





p=2
q=1
dim=4
Nsize=52
Ng=6
Ws=7
Nw=20
Wl=10


random.seed(seed_num)

walk_list1=learn_embedding(A,p,q,Nw,Wl)
walk_list2=learn_embedding(B,p,q,Nw,Wl)

trA,testyA= splitB(ya)


N=A.shape[0]
weight11=np.random.uniform(-0.1/dim,0.1/dim,(N,dim))
weight12=np.random.uniform(-0.1/dim,0.1/dim,(N,dim))

weight21=np.zeros((Nsize,dim))
weight22=np.zeros((Nsize,dim))

train_list1=skip_train(walk_list1, Ws, Ng)
train_list2=skip_train(walk_list2, Ws, Ng)

Auc1=[]
Auc2=[]
Auc3=[]

walk1 = [list(map(str, walk)) for walk in walk_list2]
model_1 = Word2Vec(walk1, size=dim, min_count=0,window=Ws,negative=Ng, sg=1, iter=40)
__, embA, outputA=MapSA(model_1, Nsize)

for iteration in range(iter_num):
    for i in range(epoc_num):
        weight11,weight12=keras_skip_gram(train_list1,weight11,weight12)
    if iteration==0:
        P=Frank_optimal(A,B,0,norm=2,mode='norm')
    else:
        D=Dmatrix(weight11,weight21,lam)
        P=Frank_optimal(A,B,D,norm=2,mode='norm')  
    weight11,weight21=update(weight11,weight21,P,0.1)
    weight12,weight22=update(weight12,weight22,P,0.1)
    for i in range(epoc_num):
        weight21,weight22=keras_skip_gram(train_list2,weight21,weight22)
    auc_l1=[]
    auc_l2=[]
    auc_l3=[]
    for c in [3**pt for pt in range(-10,10)]: 
        beta=Classifier(weight11[trA,:],ya[trA],1)
        score1=np.dot(weight11,beta)
        auc1=roc_auc_score(ya[testyA],score1[testyA])
        score2=np.dot(weight21,beta)
        auc2=roc_auc_score(yb,score2)
        score3=np.dot(embA,beta)
        auc3=roc_auc_score(yb,score3)
        auc_l1.append(auc1)
        auc_l2.append(auc2)
        auc_l3.append(auc3)
    Auc1.append(auc_l1)
    Auc2.append(auc_l2)
    Auc3.append(auc_l3)   


Cp={}
Cp['auc1']=Auc1
Cp['auc2']=Auc2
Cp['auc3']=Auc3

csts='NodeVec'+str(seed_num)+'lam'+str(lam)+'epoc'+str(epoc_num)+'iter'+str(iter_num)+'.p'
pickle.dump(Cp,open(csts,"wb"))