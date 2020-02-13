#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 23:58:43 2019

@author: yuanneu
"""

import random
import numpy as np
from keras.utils import np_utils

def splitB(Ya):
    """
    this function splits the node set 
    into the training index list and test index list
    """
    Num=Ya.shape[0]
    all_list=[i for i in range(Num)]
    train_num=int(2*Num/3)
    train_split=random.sample(all_list,train_num)
    test_list=list(set(all_list)-set(train_split))
    return train_split,test_list

def GenDirtyGraph(Num, Num_error):
    """this function generate two adjacency matrix A and B 
    both of them have Num nodes, 
    at the beginning, the matrix A and matrix B are permuation invariant.
    For graph A, the node from 0 to int(Num-1/2) is one class and the other nodes form another class
    Then randomly delete Num_error edges from each graph A and B.
    The ground truth labels for graph A and graph B are ya and yb.
    """
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
    return A,B,ya,yb,P

def GenerateM(Ya):
    """
    Generate multiple class train list and test list
    """
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