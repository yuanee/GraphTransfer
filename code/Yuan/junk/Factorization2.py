#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 23:50:11 2019

@author: yuanneu
"""

import numpy as np
import networkx as nx
import tensorflow as tf
import math
import random
import time
import node2vec
from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from Keraspackage import *
from keras.utils import np_utils
import sys
import pickle



import argparse
import numpy as np
import networkx as nx
from node2vec import *
import tensorflow as tf
import math
import random
import time
import pickle
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from keras.layers import Input, Embedding, Dense, Lambda, Reshape, Activation
from keras.models import Model
from keras import optimizers
from keras.layers.merge import concatenate, dot
from keras.layers import Input, Embedding, Dense, Lambda, Reshape, Activation
from keras.models import Model
from keras import optimizers
from keras.layers.merge import concatenate, dot
from keras import backend as K
from keras.models import Sequential
import keras


def BinaryKeras(Num,dim):
    model = Sequential()
    #model.add(Dense(3, input_dim=4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(dim, input_dim=Num, kernel_initializer='normal',use_bias=False, name='embedding',activation=None))
    model.add(Dense(1,activation='sigmoid',name='weight', kernel_initializer='zeros',  bias_initializer='zeros'))
    #model.add(Dense(1, input_dim=dim, activation='sigmoid',use_bias=False, name='weight'))
    #model.add(Dense(10, activation='softmax',name='main_input2'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #sgd=optimizers.SGD(lr=0.25, nesterov=True)
    #model.compile(loss='binary_crossentropy',
                 # optimizer=sgd,
                 # metrics=['accuracy'])
    return model
"""
def BinaryW(dim,weight):
    model = Sequential()
    model.add(Dense(3, input_dim=dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    #model.add(Dense(1, input_dim=dim, activation='sigmoid',use_bias=False, name='weight'))
    #model.add(Dense(10, activation='softmax',name='main_input2'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #sgd=optimizers.SGD(lr=0.25, nesterov=True)
    
    #model.compile(loss='binary_crossentropy',
                 # optimizer=sgd,
                 # metrics=['accuracy'])
"""
def modify_weight(model,weight):   
    model.get_layer("embedding").set_weights([weight])
    return model



def AUC(model,xtest,ytest):
    score1 = model.predict(xtest)
    auc=roc_auc_score(ytest,score1)
    return auc
   
    
def FactorGraph(Amatrix, EmbedA, lam, epsilon, Num):
    G=nx.from_numpy_matrix(Amatrix)	
    G = G.to_undirected()
    Alist=list(G.edges())
    Value=True
    num=0
    while (Value):
        random.shuffle(Alist)
        ddt=1
        EmbedAnew=EmbedA.copy()
        for (ki,kj) in Alist:
            eta=1./np.sqrt(ddt)
            z_ij=np.dot(EmbedAnew[ki,:],EmbedAnew[kj,:])
            Pij=Amatrix[ki,kj]-z_ij
            Ei=EmbedAnew[ki,:]+eta*(Pij*EmbedAnew[kj,:]-lam*EmbedAnew[ki,:])
            Ej=EmbedAnew[kj,:]+eta*(Pij*EmbedAnew[ki,:]-lam*EmbedAnew[kj,:])
            EmbedAnew[ki,:]=Ei
            EmbedAnew[kj,:]=Ej
            ddt+=1
        if (np.linalg.norm(EmbedAnew-EmbedA)**2)<=epsilon:
            Value=False
        else:
            EmbedA=EmbedAnew.copy()
        num+=1
        if num>=Num:
            Value=False
        else:
            pass
        print (np.linalg.norm(Amatrix*(Amatrix-np.dot(EmbedA,EmbedA.T)))**2+lam*np.linalg.norm(EmbedA)**2)
    return EmbedA
          


def GenerateM(Ya):
    Nclass=int(np.max(Ya))+1
    y_logit=np_utils.to_categorical(Ya, Nclass)
    train_list=[]
    inputarray=np.array([[j for j in range(Nclass)]])
    Num=Ya.shape[0]
    all_list=[i for i in range(Num)]
    train_num=int(2*Num/3)
    train_split=random.sample(all_list,train_num)
    for i in train_split:
        train_list.append([np.array([[i]]),inputarray,np.array([[y_logit[i,:]]])])
    test_list=list(set(all_list)-set(train_split))
    return train_list,test_list  


def generateB(Ya):
    Num=Ya.shape[0]
    all_list=[i for i in range(Num)]
    train_num=int(2*Num/3)
    train_split=random.sample(all_list,train_num)
    train_list=[]
    for i in range(Ya.shape[0]):
        train_list.append([np.array([[i]]),np.array([[0]]),np.array([[[ya[i]]]])])
    test_list=list(set(all_list)-set(train_split))
    return train_list,test_list


def keras_beta(trainlist,weight1,weight2):
    N,d=weight1.shape
    Nc,d=weight2.shape
    shared_layer1 = Embedding(input_dim=N, output_dim=d, weights=[weight1])
    shared_layer2 = Embedding(input_dim=Nc, output_dim=d, weights=[weight2])
    input_target = Input(shape=(1,), dtype='int32', name='input_target')
    input_negative = Input(shape=(1,),dtype='int32',name='input_beta')
    target= shared_layer1(input_target)
    beta= shared_layer2(input_negative)
    score_dot = dot([target, beta], axes=(2), normalize=False)
    sigmoid_out = Activation('sigmoid')(score_dot)
    
    model = Model(inputs=[input_target,input_negative], outputs=[sigmoid_out])
    sgd = optimizers.SGD(lr=0.025, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd)
    
    for [a1,a2,y1] in trainlist:
        loss2= model.train_on_batch([a1,a2],y1)
    embed_emb=shared_layer1.get_weights()[0]
    embed_beta=shared_layer2.get_weights()[0]
    return embed_emb,embed_beta
    
def keras_multiclass(trainlist,weight1,weight2):
    N,d=weight1.shape
    Nc,d=weight2.shape
    shared_layer1 = Embedding(input_dim=N, output_dim=d, weights=[weight1])
    shared_layer2 = Embedding(input_dim=Nc, output_dim=d, weights=[weight2])
    input_target = Input(shape=(1,), dtype='int32', name='input_target')
    input_negative = Input(shape=(Nc,),dtype='int32',name='input_beta')
    target= shared_layer1(input_target)
    beta= shared_layer2(input_negative)
    score_dot = dot([target, beta], axes=(2), normalize=False)
    sigmoid_out = Activation('softmax')(score_dot)
    
    model = Model(inputs=[input_target,input_negative], outputs=[sigmoid_out])
    sgd = optimizers.SGD(lr=0.025, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    
    for [a1,a2,y1] in trainlist:
        loss2= model.train_on_batch([a1,a2],y1)
    embed_emb=shared_layer1.get_weights()[0]
    embed_beta=shared_layer2.get_weights()[0]
    return embed_emb,embed_beta

def read_graph(Amatrix):
	'''
	Reads the input network in networkx.
	'''
	G=nx.from_numpy_matrix(Amatrix)	
	G = G.to_undirected()
	return G

def splitB(Ya):
    Num=Ya.shape[0]
    all_list=[i for i in range(Num)]
    train_num=int(2*Num/3)
    train_split=random.sample(all_list,train_num)
    test_list=list(set(all_list)-set(train_split))
    return train_split,test_list

def Dmatrix(Ea,Eb,lam):
    Nsize=Ea.shape[0]
    D=np.zeros((Nsize,Nsize))
    for i in range(Nsize):
        Ei=Ea[i,:]-Eb
        D[:,i]=np.sum(Ei*Ei,1)
    return D*lam

def Classifier(X1,Y1,C_value):
    clf=LogisticRegression(C=C_value,fit_intercept=True, penalty='l2', tol=0.01)
    clf.fit(X1,Y1)
    b1=clf.coef_[0]
    return b1  


def keras_skip_gram(trainlist1,weight1,weight2):
    N,d=weight1.shape
    negative_num=trainlist1[0][2].shape[1]
    shared_layer1 = Embedding(input_dim=N, output_dim=d, weights=[weight1])
    #shared_layer1 is the output layer
    shared_layer2 = Embedding(input_dim=N, output_dim=d, weights=[weight2])
    #shared_layer2 is the hidden layer
    input_target = Input(shape=(1,), dtype='int32', name='input_1')
    input_source = Input(shape=(1,), dtype='int32', name='input_2')
    input_negative = Input(shape=(negative_num,),dtype='int32',name='input_3')
    target= shared_layer1(input_target)
    source= shared_layer2(input_source)
    negative= shared_layer1(input_negative)
    positive_dot = dot([source, target], axes=(2), normalize=False)
    negative_dot = dot([source, negative], axes=(2), normalize=False)
    all_dot = concatenate([positive_dot, negative_dot],axis=2)
    sigmoid_sample = Activation('sigmoid')(all_dot)
    
    model = Model(inputs=[input_target,input_source,input_negative], outputs=[sigmoid_sample])
    sgd2 = optimizers.SGD(lr=0.025, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd2)
    
    for [a1,a2,a4,y1] in trainlist1:
        loss = model.train_on_batch([a1, a2, a4], y1)
    embed_output=shared_layer1.get_weights()[0]
    embed_hidden=shared_layer2.get_weights()[0]
    return embed_output,embed_hidden


def update(V,U,P,alpha):
    Vnew=(1-2*alpha)*V+2*alpha*np.dot(P,U)
    Unew=(1-2*alpha)*U+2*alpha*np.dot(P.T,V)
    return Vnew,Unew


def learn_embedding(Aa,pvalue,qvalue):
    nx_G=read_graph(Aa)
    G=Graph(nx_G,False, pvalue,qvalue)
    G.preprocess_transition_probs()
    walks=G.simulate_walks(10,10)
    return walks

def GenDirtyGraph(Num, Num_error):
    A=np.zeros((Num,Num))
    Half=int(Num/2)
    for i in range(Half):
        for j in range(i+1,Half):
            z=np.random.randint(2)
            if z==1:
                A[i,j]=1
                A[j,i]=1
    FourT=int((3*Num)/4)   
    for i in range(Half,FourT):
        for j in range(FourT,Num):
            A[i,j]=1
            A[j,i]=1
            
    for i in range(Half):
        A[i,i+Half]=1
        A[i+Half,i]=1
        
    
    
    ablist=[i for i in range(Num)]  
    random.shuffle(ablist)
    P=np.zeros((Num,Num))
    for i in range(Num):
        P[i,ablist[i]]=1
    B=np.dot(np.dot(P.T,A),P)
    ddn=0
    while (ddn<=Num_error):
        i,j=random.sample(ablist,2)
        if (B[i,j]==1):
            B[i,j]=0
            B[j,i]=0
            ddn+=1
        else:
            pass
    
    ddn=0
    while (ddn<=Num_error):
        i,j=random.sample(ablist,2)
        if (A[i,j]==1):
            A[i,j]=0
            A[j,i]=0
            ddn+=1
        else:
            pass
    
    
    ya=np.zeros((Num))
    for i in range(2):
        for j in range(Half):
            ya[Half*i+j]=i
    yb=np.dot(P.T,ya)
    return A,B,ya,yb

 
Nsize=52
dim=4 
A,B,ya,yb= GenDirtyGraph(Nsize,dim)
trainlistA,YtestA=GenerateM(ya) 
trA,testyA= splitB(ya) 


delta=0.8
Auc1=[]
Auc2=[]
Auc3=[]

embA=np.random.uniform(-1./dim,1./dim,(Nsize,dim)) 
embA=FactorGraph(A, embA, delta, 0.0001, 100)

embB=np.random.uniform(-1./dim,1./dim,(Nsize,dim)) 
embB=FactorGraph(B, embB, delta, 0.0001, 100)



Auclist=[]
input_feature=np_utils.to_categorical(np.arange(Nsize),Nsize)
xtrain=input_feature[trA,:]
xtest=input_feature[testyA]
ytrain=ya[trA]
ytest=ya[testyA]


model=BinaryKeras(Nsize,dim)
model=modify_weight(model,embA)
model.fit(xtrain, ytrain,
          epochs=1,
          batch_size=2)
lam=0.1
#score = model.evaluate(xtest, ytest, batch_size=1)
auc=AUC(model,xtest,ytest)
Auclist.append(auc)
print (auc,'auc')

Auc2=[]
Auc2new=[]
for i in range(10):
    embA=model.get_layer("embedding").get_weights()[0]
    embA=FactorGraph(A, embA, delta, 0.0001, 100)
    model=modify_weight(model,embA)
    model.fit(xtrain, ytrain,
          epochs=6,
          batch_size=2)
    auc=AUC(model,xtest,ytest) 
    print (auc,'auc',i)
    Auclist.append(auc)
    beta=model.get_layer("weight").get_weights()[0][:,0]
    score1=np.dot(embB,beta)
    auc2=roc_auc_score(yb,score1)
    Auc2.append(auc2)
    print (auc2,'auc')
embA=model.get_layer("embedding").get_weights()[0]

if iteration==0:
        P=Frank_optimal(A,B,0,norm=2,mode='norm')
else:
    D=Dmatrix(emb1,emb2,lam)
    P=Frank_optimal(A,B,D,norm=2,mode='norm') 
embB=np.zeros((Nsize,dim))        
embA,embB=update(embA,embB,P,0.1)
embB=FactorGraph(B, embB, delta, 0.0001, 100)
 
beta=model.get_layer("weight").get_weights()[0][:,0]
score2=np.dot(embB,beta)
auc2=roc_auc_score(yb,score2)
print (auc2,'auc')   
model=modify_weight(model,embA)
    

    
"""
beta=Classifier(xtrain,ytrain,1000)
score1=np.dot(xtest,beta)
auc1=roc_auc_score(ytest,score1)
"""
#print (auc1,'auc2')

#beta=model.get_layer("weight").get_weights()[0]
#score1=np.dot(xtest,beta)
#auc1=roc_auc_score(ytest,score1)

#print (auc1,'auc')



"""
emb2=np.random.uniform(-1./dim,1./dim,(Nsize,dim)) 

model=BinaryKeras(dim)
for i in range(5):
    embA=FactorGraph(B, emb2, delta, 0.0001, 100)

    for j in epoc
    emb1=np.random.uniform(-1./dim,1./dim,(Nsize,dim))   
    emb2=np.zeros((Nsize,dim)) 
    emb1=FactorGraph(A, emb1, delta, 0.0001, 100)
    
    if iteration==0:
        P=Frank_optimal(A,B,0,norm=2,mode='norm')
    else:
        D=Dmatrix(emb1,emb2,lam)
        P=Frank_optimal(A,B,D,norm=2,mode='norm') 
        
    emb1,emb2=update(emb1,emb2,P,0.1)
    emb2=FactorGraph(B, emb2, delta, 0.0001, 100)
    auc_l1=[]
    auc_l2=[]
    auc_l3=[]
    for c in [3**pt for pt in range(-10,10)]: 
        beta=Classifier(emb1[trA,:],ya[trA],c)
        score1=np.dot(emb1,beta)
        auc1=roc_auc_score(ya[testyA],score1[testyA])
        score2=np.dot(emb2,beta)
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
csts='FactorTwo'+str(ddt)+'lam'+str(lam)+'dim'+str(dim)+'iter'+str(iter_num)+'.p'
pickle.dump(Cp,open(csts,"wb"))

"""



    

    
    