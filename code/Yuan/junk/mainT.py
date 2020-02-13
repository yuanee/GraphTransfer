#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:51:08 2019

@author: yuanneu
"""

from Keraspackage import *

walk_list1=learn_embeddings(A,p,q)
walk_list2=learn_embeddings(B,p,q)


d=4
N=A.shape[0]
weight11=np.random.uniform(-1./d,1./d,(1,d))
weight12=np.random.uniform(-1./d,1./d,(N,d))
weight13=np.random.uniform(-1./d,1./d,(1,d))

weight21=np.zeros((1,d))
weight22=np.zeros((1,d))

train_list11=skip_train(walk_list1, 5, 6)
train_list12=generateL(ya)
train_list21=skip_train(walk_list2, 5, 6)

t1=time.time()
for i in range(5):
    weight11,weight12,weight13=keras_sg_first(train_list11,train_list12,weight11,weight12,weight13)


for i in range(5):
    weight21,weight22=keras_sg_second(train_list21,weight21,weight22)
    
P=Frank_optimal(A,B,0,norm=2,mode='norm')


