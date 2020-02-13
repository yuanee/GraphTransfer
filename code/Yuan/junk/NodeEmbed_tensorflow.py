#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 17:18:01 2018

@author: yuanneu
"""

import numpy as np
import networkx as nx
import tensorflow as tf
import math
import random
import time
import node2vec

def node_num(walks):
    walks=[list(np.array(a)-1) for a in walks]
    return walks

def sigmoid_fun(x):
    y=1/(1+np.exp(-x))
    return y

def softmax_fun(x_i,x_array):
    y=np.log(np.exp(x_i)/np.sum(np.exp(x_array)))
    return y

def node_fun(args,nx_graph):
    G= node2vec.Graph(nx_graph, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks= G.simulate_walks(args.num_walks, args.walk_length)
    walks=node_num(walks)
    vocabulary_size=len(nx_graph.nodes())  
    return walks,vocabulary_size

def softmax_loss_fun(encoder,decoder,walks, ylabel):     
    cross_dot=np.dot(encoder,encoder.T)
    loss_softmax=0
    for walk_input in walks:
        u1=walk_input[0]
        for u2 in walk_input[1::]:
            loss_softmax+=softmax_fun(cross_dot[u1,u2],cross_dot[u1,:])
    y_out=np.mean(sigmoid_fun(np.dot(encoder,decoder)),1)
    loss_label=np.sum(np.square(y_out-ylabel))
    loss_total=loss_softmax+loss_label
    return loss_total

def NodeEmbedding(args):
    walks,vocabulary_size=node_fun(args,args.nx_graph)
    walk_size=args.walk_length-1
    
    
    walk_initial = tf.placeholder(dtype=tf.int32, shape=[None])
    y_true = tf.placeholder(dtype=tf.float32, shape=[None])
    train_input = tf.placeholder(tf.int32, shape=[walk_size])
    train_context = tf.placeholder(tf.int32, shape=[walk_size, 1])
    
    embed_W=tf.Variable(tf.random_uniform([vocabulary_size, args.embedding_size]),[-1.0,1.0])
    decoder_W=tf.Variable(tf.random_uniform([args.embedding_size,args.output_size]),[-1.0,1.0])
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, args.embedding_size],
                            stddev=1.0 / math.sqrt(args.embedding_size)))
    embed=tf.nn.embedding_lookup(embed_W, train_input)
    v_i= tf.nn.embedding_lookup(embed_W, walk_initial)
    decoder_y=tf.matmul(v_i,decoder_W)
    y_sigmoid=tf.sigmoid(decoder_y)
    
    nce_loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,biases=nce_biases,\
                labels=train_context,inputs=embed,num_sampled=args.num_sam, num_classes=vocabulary_size))        
    loss_label=tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_sigmoid,1)))
    l2_dis=tf.reduce_sum(tf.square(embed_W-nce_weights))
    
    loss=loss_label+l2_dis+nce_loss
    
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
    init = tf.global_variables_initializer()    
        
    with tf.Session() as sess:
        sess.run(init)
        for ep in range(args.iter):  # epochs loop
            for batch_n in range(args.walk_length):
                walk_input=walks[batch_n]
                batch_input=[walk_input[0]]*(len(walk_input)-1)
                batch_context=np.array([walk_input[1:]]).T 
                begin_point=[walk_input[0]]
                y_input=args.ylabel[begin_point]              
                sess.run(train_op, feed_dict={train_input: batch_input, train_context: batch_context,\
                                          walk_initial:begin_point, y_true: y_input})
             
            [encoder,decoder]=sess.run([embed_W,decoder_W])
            
            loss_total=softmax_loss_fun(encoder,decoder,walks, args.ylabel)
            print (loss_total,ep)
    
    
