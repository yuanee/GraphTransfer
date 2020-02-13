#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 13:58:41 2019

@author: yuanneu
"""


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
from Pfunpackage import *




from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import random




def node_num(walks):
    walks=[list(np.array(a)-1) for a in walks]
    return walks 
  
def frequency(walks):
    P_m={}
    for walk in walks:
        for item in walk:
            try: 
                P_m[item]+=1
            except:
                P_m[item]=1
    for key, value in P_m.items():
        P_m[key]=value**0.75
    return P_m

def negative_frequency(P_m):
    sample_num=[]
    sample_prob=[]
    for key, value in P_m.items(): 
        sample_num.append(key)
        sample_prob.append(value)
    return sample_num, np.array(sample_prob)/sum(sample_prob)

def get_negative_sample(context,num,prob,Gn):
    negative_list=[]    
    while len(negative_list)<Gn:
        negative_sample = np.random.choice(num, p= prob.ravel())
        if (negative_sample!=context):
            negative_list.append(negative_sample)
        else:
            pass
    return np.array([negative_list])  

def skip_train(walks, window_size, Gn):
    """
    use the wallks to generate negative samples for neural network
    """
    P_m=frequency(walks)
    Num,Prob=negative_frequency(P_m)
    p_dict=frequency(walks)        
    num, prob=negative_frequency(p_dict)    
    train_list=[]    
    for walk in walks:        
        for pod, word in enumerate(walk):  
            reduced_window = np.random.randint(window_size)
            source_input=np.array([[word]])
            start = max(0, pod - window_size+reduced_window)
            for pod2 in range(start, pod+window_size+1-reduced_window):
                #print (pod2,pod)
                if pod2!=pod:
                    try: 
                        target_input=np.array([[walk[pod2]]])
                        negative_input=get_negative_sample(target_input,Num,Prob,Gn)                        
                        output_label=np.concatenate((np.array([[1.0]]), np.zeros((1,Gn))),1)                         
                        train_list.append([target_input, source_input, negative_input, np.array([output_label])])
                    except:
                        pass
                else:
                    pass
    return train_list

def read_graph(Amatrix):
	'''
	Reads the input network in networkx.
	'''
	G=nx.from_numpy_matrix(Amatrix)	
	G = G.to_undirected()
	return G


def learn_embedding(Aa,pvalue,qvalue,num_walk,walk_length):
    nx_G=read_graph(Aa)
    G=Graph(nx_G,False, pvalue,qvalue)
    G.preprocess_transition_probs()
    walks=G.simulate_walks(num_walk,walk_length)
    return walks

def update(V,U,P,alpha):
    Vnew=(1-2*alpha)*V+2*alpha*np.dot(P,U)
    Unew=(1-2*alpha)*U+2*alpha*np.dot(P.T,V)
    return Vnew,Unew

def MapSA(model, Num):
    Ma=np.zeros((Num,Num))
    Xa=model.wv.vectors
    Xa_out=model.syn1neg
    featureA=[]
    for key in range(Num):
        featureA.append(model[str(key)])
    featureA=np.array(featureA)
    for ki in range(Num):
        for kj in range(Num):
            zt=np.linalg.norm(Xa[ki,:]-featureA[kj,:])
            if zt<=1e-6:
                Ma[ki,kj]=1
            else:
                pass
    output_weight=np.dot(Ma.T,Xa_out)
    return Ma, featureA, output_weight

def Dmatrix(Ea,Eb,lam):
    Nsize=Ea.shape[0]
    D=np.zeros((Nsize,Nsize))
    for i in range(Nsize):
        Ei=Ea[i,:]-Eb
        D[:,i]=np.sum(Ei*Ei,1)
    return D*lam

def splitB(Ya):
    Num=Ya.shape[0]
    all_list=[i for i in range(Num)]
    train_num=int(2*Num/3)
    train_split=random.sample(all_list,train_num)
    test_list=list(set(all_list)-set(train_split))
    return train_split,test_list

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

def Classifier(X1,Y1,C_value):
    clf=LogisticRegression(C=C_value,fit_intercept=True, penalty='l2', tol=0.01)
    clf.fit(X1,Y1)
    b1=clf.coef_[0]
    return b1   



