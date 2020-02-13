#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 19:00:13 2019

@author: yuanneu
"""



import numpy as np
import math
import random

from cvxopt import matrix
import numpy as np
from cvxopt import solvers
import time
import pickle
from scipy.optimize import linear_sum_assignment
import sys


def size_determine(Aa,Ab,Ac,Ad):
    if Ad**2>=Ac:
        if Ad>=0:
            size=0.
        else:
            size=1.
    else:
        lam0=Ab/Ac
        C0=Aa-Ab**2/Ac
        delta_c=C0*Ad**2/(Ac**2-Ad**2*Ac)
        eta1=-lam0+np.sqrt(delta_c)
        eta2=-lam0-np.sqrt(delta_c)
        etalist2=[0.,1.,eta1,eta2]
        etalist=[eta for eta in etalist2 if (eta<=1)&(eta>=0)]
        fun=lambda eta: np.sqrt(Ac*eta**2+2*Ab*eta+Aa)+Ad*eta
        ylist=[fun(eta) for eta in etalist]
        size=etalist[np.argmin(ylist)]
    return size

def V_dot(v):
    """
    the ADMM algorithm to solve:
    min <S, V>
    subject to S>=0, S1=1, 1^{T}S=1^{T}
    """
    row_ind, col_ind = linear_sum_assignment(v)
    b=np.zeros(v.shape)
    b[row_ind,col_ind]=1
    return b

def delete_p(array1,array2,i,j):
    #array2[j]+=array1[i]
    arr1new=np.delete(array1,i)
    con_d=np.concatenate((arr1new,array2))
    return con_d.copy()


def descent_norm(A,B,D,P):
    """
    the gradient descent for ||AP-PB||_{2}+tr(P^{T}D)
    """
    C=np.dot(A,P)-np.dot(P,B)
    Y=(np.dot(A.T,C)-np.dot(C,B.T))/np.linalg.norm(C)+D
    return Y
       
def deriveP(A,B,D,P,order):
    Deriv=np.zeros(A.shape)
    Cmatrix=np.dot(A,P)-np.dot(P,B)
    row,column=np.shape(A)
    total_p=np.sum(Cmatrix**order)
    total_norm=total_p**(1./order)
    for i in range(row):
        for j in range(column):
            P_old=P.copy()
            P_old[i,j]+=0.001
            DotRow=np.dot(A,P_old[:,j])-np.dot(P_old,B[:,j])
            DotColumn=np.dot(A[i,:],P_old)-np.dot(P_old[i,:],B)
            mat1=delete_p(DotRow,DotColumn,i,j)
            ar1,ar2=Cmatrix[:,j].copy(),Cmatrix[i,:].copy()
            mat2=delete_p(ar1,ar2,i,j)
            diff=np.sum(mat1**order)-np.sum(mat2**order)
            deriv=((total_p+diff)**(1./order)-total_norm)/0.001
            Deriv[i,j]=deriv
    Deriv+=D
    return Deriv
            

def loss_p_norm(A,B,D,P,order):
    C=np.dot(A,P)-np.dot(P,B)
    c_p=(np.sum(C**order))**(1./order)
    pd=np.sum(P*D)
    return (c_p+pd)

def norm_loss_fun(a,b,d,p):
    """
    ||AP-PB||_{2}+tr(P^{T}D)
    """
    c=np.dot(a,p)-np.dot(p,b)
    loss=np.linalg.norm(c)+np.trace(np.dot(p.T,d))
    return loss
    
def Frank_wolfe_p(A,B,D,epsilon,order,itr_num,mode):
    Loss=[]
    Time=[]
    Dual=[]
    loss=10**10
    P=np.identity(A.shape[0])
    itr=0
    while(True):
        t1=time.time()
        nabla_P=deriveP(A,B,D,P,order)
        S_optimal=V_dot(nabla_P)
        delta_P=S_optimal-P  
        if mode==1:
            eta=2/(itr+2)
        else:
            Cm0=np.dot(A,P)-np.dot(P,B)
            Cm1=np.dot(A,delta_P)-np.dot(delta_P,B)
            Aa=np.sum(Cm0*Cm0)
            Ab=np.sum(Cm1*Cm0)
            Ac=np.sum(Cm1*Cm1)
            Ad=np.sum(delta_P*D)
            
            eta=size_determine(Aa,Ab,Ac,Ad)
        dual=-np.sum(nabla_P*delta_P)
        Dual.append(dual)
        P_new=P+eta*delta_P
        loss=loss_p_norm(A,B,D,P_new,order)
        print (itr,loss,dual)
        Loss.append(loss)
        P=P_new.copy()
        t2=time.time()
        Time.append(t2-t1)
        itr+=1
        if itr>=itr_num:
            break
        else:
            pass
        if dual<=epsilon:
            break
        else:
            pass
    return P_new,Loss,Dual,Time


timefa1=float(sys.argv[1])
ddt=int(timefa1)
covs=float(sys.argv[2])
hyper=1*covs
fold=float(sys.argv[3])
fold=int(fold)    

N=ddt
order=hyper
see=fold      



np.random.seed(see)
a=np.array(np.random.randint(0,2,(N,N)),dtype=float)
b=np.array(np.random.randint(0,2,(N,N)),dtype=float)
d=np.random.normal(np.zeros(N), np.ones((N,N)))
#p=np.random.normal(np.zeros(N), np.ones((N,N)))

#c=derive(a,b,d,p,4.0)
#c1=deriveP(a,b,d,p,4.0)

ct,Loss,Dual,Time=Frank_wolfe_p(a,b,d,0.001,order,10000,1)
Cp={}
Cp['frank']=[ct,Loss,Dual,Time]
csts='FrankNorm'+str(order)+'P'+str(N)+'see'+str(see)+'.p'
pickle.dump(Cp,open(csts,"wb"))
