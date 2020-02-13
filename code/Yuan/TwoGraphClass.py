#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 11:16:17 2019

@author: yuanneu
"""
from Node2vecUpdate import *
Nsize=48
dim=16

A,B,ya,yb,trueP = GenDirtyGraph(Nsize,dim)


returnP=0.25
outputP=4
num_walk=10
walk_length=40
window_size=5
negative_num=4
dim=4
iteration=5
walks=learn_embedding(A,returnP,outputP,num_walk,walk_length)
trainlist=skip_train(walks,window_size,negative_num,iteration)

walksB=learn_embedding(B,returnP,outputP,num_walk,walk_length)
trainlistB=skip_train(walksB,window_size,negative_num,iteration)



#emb1,emb2=keras_sg_embedding(trainlist,weight1,weight2)

model,wa=word2vec(walks,window_size, negative_num, dim, iteration)
Xfeature=[]
for n in range(Nsize):
    Xfeature.append(model.wv[str(n)]/np.linalg.norm(model.wv[str(n)]))
    for j in range(Nsize):
        print(n,j, model.wv.similarity(str(n),str(j)))
Xa=np.array(Xfeature)

weight1=np.random.multivariate_normal(np.zeros(dim), 0.1*np.identity(dim), Nsize)
weight2=np.random.multivariate_normal(np.zeros(dim), 0.1*np.identity(dim), Nsize)

weight1,weight2=keras_sg_embedding(trainlist,weight1,weight2)

#Xfeature2=[]
#for n in range(Nsize):
#    Xfeature2.append(weight2[n,:]/np.linalg.norm(weight2[n,:]))
#Xa2=np.array(Xfeature2)

train_index=random.sample(range(Nsize),30)
test_index=list(set(range(Nsize))-set(train_index))
Xtrain=weight2[train_index,:]
Xtest=weight2[test_index,:]
Ytrain=ya[train_index]
Ytest=ya[test_index]
clf=LogisticRegression(C=1,fit_intercept=True, penalty='l2', tol=0.01)
clf.fit(Xtrain,Ytrain)
b1=clf.coef_[0]
score1=np.dot(Xtest, b1)
auc=roc_auc_score(Ytest,score1)

for itera in range(1):
    weightBT1=np.zeros((Nsize,dim))
    weightBT2=np.zeros((Nsize,dim))
    
    #for j in epoc
    
    if itera==0:
        P=Frank_optimal(A,B,0,norm=2,mode='norm')
    else:
        P=Frank_optimal(A,B,D,norm=2,mode='norm') 
    
    score2S=np.dot(np.dot(P.T,weightBT2), b1)
    auc2S=roc_auc_score(yb,score2S)
    
    weight1, weightBT1= update(weight1, weightBT1,P,0.1)   
    weight2, weightBT2= update(weight2, weightBT2,P,0.1)  
    weightBT1,weightBT2=keras_sg_embedding(trainlistB,weightBT1,weightBT2)
    
    ### The classifier Train
    #weightBT1,weightBT2=keras_sg_embedding(trainlistB,weightBT1,weightBT2)
    
    score2=np.dot(weightBT2, b1)
    auc2=roc_auc_score(yb,score2)
