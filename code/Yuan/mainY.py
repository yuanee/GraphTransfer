#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:31:09 2019

@author: yuanneu
"""

from Node2vecUpdate import *
import argparse


def update(V,U,P,alpha):
    Vnew=(1-2*alpha)*V+2*alpha*np.dot(P,U)
    Unew=(1-2*alpha)*U+2*alpha*np.dot(P.T,V)
    return Vnew,Unew

if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--N',type=int,default=26,help='Number of node')
    parser.add_argument('--dim',type=float,default=4,help='Dimension of embedding')
    parser.add_argument('--p', type=float, default=0.25,
    	                    help='Return hyperparameter. Default is 1.')
    parser.add_argument('--q', type=float, default=4.,
    	                    help='Inout hyperparameter. Default is 1.')
    parser.add_argument('--num', type=int, default=10,
    	                    help='number of walk')
    parser.add_argument('--length', type=int, default=40,
    	                    help='walk length')
    parser.add_argument('--size', type=int, default=5,
    	                    help='Context size for optimization')
    parser.add_argument('--negative', type=int, default=4,
    	                    help='Negative sample number')
    parser.add_argument('--iter', type=int, default=20,
    	                    help='Iteration number')
    args = parser.parse_args()
    
    A,B,ya,yb= GenDirtyGraph(args.N,args.dim)
    trainlistA,YtestA=GenerateM(ya) 
    trA,testyA= splitB(ya) 
    G=read_graph(A)
    walks=learn_embedding(A,args.p,args.q,args.num,args.length)
    model=word2vec(walks,args.size, args.negative, args.dim, args.iter)
    trainlist=skip_train(walks,args.size,args.negative,args.iter)
    weight1=np.random.multivariate_normal(np.zeros(args.dim), 0.1*np.identity(args.dim), args.N)
    weight2=np.random.multivariate_normal(np.zeros(args.dim), 0.1*np.identity(args.dim), args.N)
    weight1,weight2=keras_sg_embedding(trainlist,weight1,weight2)
    
    
    
    
    weightB1=np.random.multivariate_normal(np.zeros(args.dim), 0.1*np.identity(args.dim), args.N)
    weightB2=np.random.multivariate_normal(np.zeros(args.dim), 0.1*np.identity(args.dim), args.N)
    walksB=learn_embedding(B,args.p,args.q,args.num,args.length)
    trainlistB=skip_train(walks,args.size,args.negative,args.iter)
    weightB1,weightB2=keras_sg_embedding(trainlistB,weightB1,weightB2)
    
    
    for iteration in range(1):
        weightBT1=np.zeros(args.N,args.dim)
        weightBT2=np.zeros(args.N,args.dim)
        
        #for j in epoc
        
        if iteration==0:
            P=Frank_optimal(A,B,0,norm=2,mode='norm')
        else:
            D=Dmatrix(emb1,emb2,lam)
            P=Frank_optimal(A,B,D,norm=2,mode='norm') 
         
        weight1, weightBT1= update(weight1, weightBT1,P,0.1)   
        weight2, weightBT2= update(weight2, weightBT2,P,0.1)  
        weightBT1,weightBT2=keras_sg_embedding(trainlistB,weightBT1,weightBT2)
        
        ### The classifier Train
        weightBT1,weightBT2=keras_sg_embedding(trainlistB,weightBT1,weightBT2)
        D=np.zeros((args.dim,args.dim))
        for i in range(args.dim):
            for j in range(args.dim):
                D[i,j]=np.dot(weight1[i,:]-weightBT1[j,:],weight1[i,:]-weightBT1[j,:])
        

    
    
    
    
    
    
   
   






