#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:15:52 2019

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
#from Factorization import *
#from NodeEmbed import *
import matplotlib.pyplot as plt 
from gensim.models import Word2Vec

from keras.layers import Input, Embedding, Dense, Lambda, Reshape, Activation
from keras.models import Model
from keras import optimizers
from keras.layers.merge import concatenate, dot
from Pfunpackage import *

from Transfer import *


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

def skip_train(walks, window_size):
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

def generateL(ya):
    Train_list=[]
    for i in range(len(ya)):
        Train_list.append([np.array([[i]]),np.array([[0]]),np.array([[[ya[i]]]])])
    #for i in range(8):
        #Train_list.append([np.array([[i+14]]),np.array([[0]]),np.array([[[0]]])])
    return 3*Train_list


def keras_sg_first(trainlist1,trainlist2,weight1,weight2,weight3):
    N,d=weight1.shape
    negative_num=trainlist1[0][2].shape[1]
    shared_layer1 = Embedding(input_dim=N, output_dim=d, weights=[weight1])
    #shared_layer1 is the output layer
    shared_layer2 = Embedding(input_dim=N, output_dim=d, weights=[weight2])
    #shared_layer2 is the hidden layer
    shared_layer3 = Embedding(input_dim=1, output_dim=d, weights=[weight3])
    #shared_layer3 is the classifier layer
    input_target = Input(shape=(1,), dtype='int32', name='input_1')
    input_source = Input(shape=(1,), dtype='int32', name='input_2')
    input_negative = Input(shape=(negative_num,),dtype='int32',name='input_3')
    input_beta = Input(shape=(1,),dtype='int32', name='input_beta')
    target= shared_layer1(input_target)
    source= shared_layer2(input_source)
    negative= shared_layer1(input_negative)
    beta= shared_layer3(input_beta)
    positive_dot = dot([source, target], axes=(2), normalize=False)
    negative_dot = dot([source, negative], axes=(2), normalize=False)
    score_dot = dot([source, beta], axes=(2), normalize=False)
    all_dot = concatenate([positive_dot, negative_dot],axis=2)
    sigmoid_sample = Activation('sigmoid')(all_dot)
    sigmoid_out = Activation('sigmoid')(score_dot)
    
    model = Model(inputs=[input_target,input_source,input_negative], outputs=[sigmoid_sample])
    sgd2 = optimizers.SGD(lr=0.025, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd2)
    model2 = Model(inputs=[input_source,input_beta], outputs=[sigmoid_out])
    model2.compile(loss='binary_crossentropy', optimizer=sgd2)
    for [a1,a2,a4,y1] in trainlist1:
        loss = model.train_on_batch([a1, a2, a4], y1)
    for [a1,a2,y1] in trainlist2:
        loss2= model2.train_on_batch([a1,a2],y1)
    embed_output=shared_layer1.get_weights()[0]
    embed_hidden=shared_layer2.get_weights()[0]
    embed_parameter=shared_layer3.get_weights()[0]
    return embed_output,embed_hidden,embed_parameter



def update(V,U,P,alpha):
    Vnew=(1-2*alpha)*V+2*alpha*np.dot(P,U)
    Unew=(1-2*alpha)*U+2*alpha*np.dot(P.T,V)
    return Vnew,Unew


def read_graph(Amatrix):
	'''
	Reads the input network in networkx.
	'''
	G=nx.from_numpy_matrix(Amatrix)	
	G = G.to_undirected()
	return G


def learn_embedding(Aa,pvalue,qvalue):
    nx_G=read_graph(Aa)
    G=Graph(nx_G,False, pvalue,qvalue)
    G.preprocess_transition_probs()
    walks=G.simulate_walks(10,10)
    print (len(walks))
    return walks

def update(V,U,P,alpha):
    Vnew=(1-2*alpha)*V+2*alpha*np.dot(P,U)
    Unew=(1-2*alpha)*U+2*alpha*np.dot(P.T,V)
    return Vnew,Unew
