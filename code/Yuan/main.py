#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:31:09 2019

@author: yuanneu
"""

import argparse
import time
import numpy as np

from ClassDataGen import GenDirtyGraph
from Node2vecUpdate import learn_embedding, word2vec, skip_train, keras_sg_embedding


def node2vec_similarity_matrices(nepochs, n_embedding, n_walks, walk_length, window_size, p, q, n_negative, A=None):
#if __name__ == "__main__":
#    
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--N',type=int,default=26, help='Number of node')
#    parser.add_argument('--dim',type=float,default=4, help='Dimension of embedding')
#    parser.add_argument('--p', type=float, default=0.25, help='Return hyperparameter. Default is 1.')
#    parser.add_argument('--q', type=float, default=4., help='Inout hyperparameter. Default is 1.')
#    parser.add_argument('--num', type=int, default=10, help='number of walk')
#    parser.add_argument('--length', type=int, default=40, help='walk length')
#    parser.add_argument('--size', type=int, default=5, help='Context size for optimization')
#    parser.add_argument('--negative', type=int, default=4, help='Negative sample number')
#    parser.add_argument('--iter', type=int, default=20, help='Iteration number')
#    args = parser.parse_args()
    
    args = argparse.Namespace()
    args.dim = n_embedding #5
    args.p = p #0.25
    args.q = q #4
    args.num = n_walks #20
    args.length = walk_length #10
    args.size = window_size #4
    args.negative = n_negative #5
    args.iter = nepochs
    
    starttime = time.clock()
    if A is None:
        A,B,ya,yb= GenDirtyGraph(26,args.dim)
    args.N = A.shape[0]
    print("Generate walks")
    starttime = time.clock()
    walks=learn_embedding(A,args.p,args.q,args.num,args.length)
    print("Time elapsed = %.4f" % (time.clock() - starttime))
    
    print("Run gensim node2vec")
    starttime = time.clock()
    model=word2vec(walks,args.size, args.negative, args.dim, args.iter)
    print("Time elapsed = %.4f" % (time.clock() - starttime))
    Xfeature1=[]
    for n in range(args.N):
        Xfeature1.append(model.wv[str(n)]/np.linalg.norm(model.wv[str(n)]))
    Xa1=np.array(Xfeature1)
    Sa1=np.zeros((args.N,args.N))
    
    print("Generate train list for node2vec")
    starttime = time.clock()
    trainlist=skip_train(walks,args.size,args.negative,args.iter)
    print("Time elapsed = %.4f" % (time.clock() - starttime))
    
    print("Run Keras node2vec")
    starttime = time.clock()
    weight1=np.random.multivariate_normal(np.zeros(args.dim), 0.1*np.identity(args.dim), args.N)
    weight2=np.random.multivariate_normal(np.zeros(args.dim), 0.1*np.identity(args.dim), args.N)
    weight11, weight12 = keras_sg_embedding(trainlist,weight1,weight2)
    print("Time elapsed = %.4f" % (time.clock() - starttime))
    
    Xfeature21=[]
    for n in range(args.N):
        Xfeature21.append(weight12[n,:]/np.linalg.norm(weight12[n,:]))
    Xa21=np.array(Xfeature21)
    Sa21=np.zeros((args.N,args.N))
    
    for i in range(args.N):
        for j in range(args.N):
            Sa1[i,j]=np.dot(Xa1[i,:],Xa1[j,:])
            Sa21[i,j]=np.dot(Xa21[i,:],Xa21[j,:])
    D11=np.linalg.norm(Sa1-Sa21)**2/(np.linalg.norm(Sa1)**2+np.linalg.norm(Sa21)**2)
    
    print("Difference between\n\tgensim & Yuan's node2vec: %.6f" % (D11))
    
    return weight1, weight2, Xa1, Xa21, Sa1, Sa21, D11
    
    
    






