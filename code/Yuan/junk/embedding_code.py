#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 15:44:30 2018

@author: yuanneu
"""


import argparse
import numpy as np
import networkx as nx
import node2vec
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



def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")
    
    parser.add_argument('--input', nargs='?', default='/Users/yuanneu/Downloads/node2vec-master/graph/karate.edgelist',
                       help='Input graph path')
    
    parser.add_argument('--output', nargs='?', default='/Users/yuanneu/Downloads/node2vec-master/graph/karate.emb',
                       help='Embeddings path')
    
    parser.add_argument('--dimensions', type=int, default=10,
                       help='Number of dimensions. Default is 128.')
    
    parser.add_argument('--walk-length', type=int, default=30,
                       help='Length of walk per source. Default is 80.')
    
    parser.add_argument('--num-walks', type=int, default=5,
                       help='Number of walks per source. Default is 10.')
        
    parser.add_argument('--embedding-size', type=int, default=10,
                       help='dimension of embedding. Default is 10.')
     
    parser.add_argument('--num_sam', type=int, default=10,
                       help='negative sampling number. Default is 10.')
        
    #parser.add_argument('--output-size', type=int, default=10,
                       #help='dimension of decoder. Default is 4.')
        
    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')
    
    parser.add_argument('--iter', default=1, type=int,
                          help='Number of epochs in SGD')
    
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of parallel workers. Default is 8.')
    
    parser.add_argument('--p', type=float, default=1,
                       help='Return hyperparameter. Default is 1.')
    
    parser.add_argument('--q', type=float, default=2,
                       help='Inout hyperparameter. Default is 1.')
    
    parser.add_argument('--weighted', dest='weighted', action='store_true',
                       help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)
    
    parser.add_argument('--directed', dest='directed', action='store_true',
                       help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()

def read_graph():
	'''
	Reads the input network in networkx.
	'''
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1
	if not args.directed:
		G = G.to_undirected()
	return G


def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [map(str, walk) for walk in walks]
	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	model.save_word2vec_format(args.output)	
	return

def node_num(walks):
    walks=[list(np.array(a)-1) for a in walks]
    return walks 
  
  
class graph_value_G():
    def __init__(self, realpart, vocabulary, vocabulary_list,embedding_size,window_size):
        self.negative = realpart
        self.vocabulary=vocabulary
        self.vocabulary_list = vocabulary_list
        self.embedding_size=embedding_size
        self.window_size=window_size 

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

def get_negative_sample(context,num,prob):
    negative_list=[]    
    while len(negative_list)<G.negative:
        negative_sample = np.random.choice(num, p= prob.ravel())
        if (negative_sample!=context):
            negative_list.append(negative_sample)
        else:
            pass
    return np.array([negative_list])  

def skip_train(walks, window_size):
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
                if pod2!=pod:
                    try: 
                        target_input=np.array([[walk[pod2]]])
                        negative_input=get_negative_sample(target_input,Num,Prob)
                        output_label=np.concatenate((np.array([[1.0]]), np.zeros((1,G.negative))),1) 
                        train_list.append([target_input, source_input, negative_input, np.array([output_label])])
                    except:
                        pass
                else:
                    pass
    return train_list


def Keras_skip_gram(G,walks,iteration):  
    """
    Keras to run word2vec algorithm with skip_gram model.
    """

    walks_sentences=[list(np.array(walk)) for walk in walks]
               
    embedding1=np.random.uniform(-1/G.embedding_size,1/G.embedding_size,(G.vocabulary,G.embedding_size))
    embedding2=np.random.uniform(-1/G.embedding_size,1/G.embedding_size,(G.vocabulary,G.embedding_size))
    shared_layer1 = Embedding(input_dim=G.vocabulary, output_dim=G.embedding_size, weights=[embedding1])
    shared_layer2 = Embedding(input_dim=G.vocabulary, output_dim=G.embedding_size, weights=[embedding2])
    
    input_target = Input(shape=(1,), dtype='int32', name='input_1')
    input_source = Input(shape=(1,), dtype='int32', name='input_2')
    input_negative = Input(shape=(G.negative,),dtype='int32',name='input_3')
    
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
    
    train_list=skip_train(walks_sentences,G.window_size)
    
    for i in range(iteration):
        for [a1,a2,a4,y1] in train_list:
            loss = model.train_on_batch([a1, a2, a4], y1)
    embed=shared_layer2.get_weights()[0]
    
    return embed



          
            
