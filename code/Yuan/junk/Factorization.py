#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 17:11:38 2018

@author: yuanneu
"""


import numpy as np
import networkx as nx
import tensorflow as tf
import math
import random
import time


def sigmoid_fun(x):
    y=1/(1+np.exp(-x))
    return y

def factorization_graph(nx_graph):
    adj_mat=np.array(nx.adjacency_matrix(nx_graph).todense())
    edge_dict={}
    for node in nx_graph.nodes():
        edge_dict[node-1]=[i-1 for i in list(nx_graph.neighbors(node))]
    node_list=list(edge_dict.keys())  
    vocabulary_size=len(node_list)
    return adj_mat,edge_dict,node_list,vocabulary_size

def factorizaiton_loss(encoder,decoder,adj_mat,y_array):
    sim_out=np.dot(encoder,encoder.T)
    y_out=np.mean(sigmoid_fun(np.dot(encoder,decoder)),1)
    loss_sim=np.sum(np.square(sim_out-adj_mat)*adj_mat)
    loss_label=np.sum(np.square(y_out-y_array))
    loss_total=loss_sim+loss_label
    return loss_total

def FactorizationT(args):  
    
    adj_mat,edge_dict,node_list,vocabulary_size=factorization_graph(args.nx_graph)
       
    args.embedding_size=10
    args.output_size=4
    
    inputs = tf.placeholder(dtype=tf.int32, shape=[None])
    input_edge=tf.placeholder(dtype=tf.int32, shape=[None])
    y_true = tf.placeholder(dtype=tf.float32, shape=[None])
    sim = tf.placeholder(dtype=tf.float32, shape=[1,None])
    
    embed_W=tf.Variable(tf.random_uniform([vocabulary_size, args.embedding_size]),[-1.0,1.0])
    decoder_W=tf.Variable(tf.random_uniform([args.embedding_size,args.output_size]),[-1.0,1.0])
    
    v_i= tf.nn.embedding_lookup(embed_W, inputs)
    vs_edge=tf.nn.embedding_lookup(embed_W,input_edge)
    decoder_y=tf.matmul(v_i,decoder_W)
    y_sigmoid=tf.sigmoid(decoder_y)
    sim_out=tf.matmul(v_i, vs_edge, transpose_b=True)
        
    loss_label=tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_sigmoid,1)))
    loss_sim=tf.reduce_sum(tf.square(sim_out-sim))
    loss = loss_sim+loss_label
    
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
    init = tf.global_variables_initializer()    
    
    with tf.Session() as sess:
        sess.run(init)
        for ep in range(args.iter):  # epochs loop
            for batch_n in range(args.walk_length):  # batches loop            
                node_input=random.sample(node_list,1)
                node=node_input[0]
                node_edge=edge_dict[node]
                sim_input=np.array([adj_mat[node,:][node_edge]])
                y_input=args.ylabel[node_input]
                sess.run(train_op, feed_dict={inputs: node_input, input_edge:node_edge, y_true: y_input, sim: sim_input})
                
            [encoder,decoder]=sess.run([embed_W,decoder_W])    
            loss_out=factorizaiton_loss(encoder,decoder,adj_mat,args.ylabel)
            print (ep, loss_out)
    return encoder,decoder
