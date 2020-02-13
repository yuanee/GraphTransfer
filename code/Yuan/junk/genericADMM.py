#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:48:53 2019

@author: yuanneu
"""

import numpy as np
import math
import random

from cvxopt import matrix
from cvxopt import solvers
import time
import pickle



def projectToVA(x,A,r):
    Ac=set(x.keys()).difference(set(A))
    offset=1.0/(len(Ac))*(sum([x[i] for i in Ac])-r)
    y=dict([(i,0.0) for i in A]+[(i,x[i]-offset) for i in Ac])
    return y 

def projectToPositiveSimplex(x,r):
    """ A function that projects a vector x to the face of the positive simplex. 
	Given x as input, where x is a dictionary, and a r>0,  the algorithm returns a dictionary y with the same keys as x such that:
        (1) sum( [ y[key] for key in y] ) == r,
        (2) y[key]>=0 for all key in y
        (3) y is the closest vector to x in the l2 norm that satifies (1) and (2)  
        The algorithm terminates in at most O(len(x)) steps, and is described in:
	 
             Michelot, Christian. "A finite algorithm for finding the projection of a point onto the canonical simplex of R^n." Journal of Optimization Theory and Applications 50.1 (1986): 195-200iii
        and a short summary can be found in Appendix C of:
 
	     http://www.ece.neu.edu/fac-ece/ioannidis/static/pdf/2010/CR-PRL-2009-07-0001.pdf
    """ 
    A=[]
    y=projectToVA(x,A,r)
    B=[i for i in y.keys() if y[i]<0.0]
    while len(B)>0:
        A=A+B
        y=projectToVA(y,A,r)
        B=[i for i in y.keys() if y[i]<0.0]
    return y    

def VecSimplex(x,r):
    """
    max ||s-v||_{2}^{2}
    subject to <s,1>=r, s>=0
    """
    size=len(x)
    xd={}
    for i in range(size):
        xd[i]=x[i]
    y=projectToPositiveSimplex(xd,r)
    return np.array([y[i] for i in range(size)])    


def Xupdate(Z,Y,rho):
    V=Z-Y/rho
    Xtrace=np.zeros(V.shape)
    for i in range(V.shape[1]):
        Xtrace[:,i]=VecSimplex(V[:,i],1)
    return Xtrace


def Psquare(A,B,Y,Z,D,rho,i,j):
    E=np.zeros((A.shape))
    E[:,j]=A[i,:]
    E[i,:]=-B[:,j]
    E[i,j]=A[i,i]-B[j,j]
    V=Z-Y/rho
    V[i,j]-=(D[i,j]/rho)
    C=np.sum(V*E)/(1+2./rho*np.sum(E*E))
    Pnew=V-2*C*E/rho
    return Pnew

def Xupdate2(Z,Y,rho):
    V=Z-Y/rho
    Xtrace=np.zeros(V.shape)
    for i in range(V.shape[0]):
        Xtrace[i,:]=VecSimplex(V[i,:],1)
    return Xtrace

def Zupdate(Xdict,Ydict,rho):
    Xmean=np.mean(np.array(list(Xdict.values())),0)
    Ymean=np.mean(np.array(list(Ydict.values())),0)
    V=Xmean+Ymean/rho
    #print (V,'V')
    Znew=np.zeros(V.shape)
    for i in range(V.shape[0]):
        #print (V[i,:],'v')
        Znew[i,:]=VecSimplex(V[i,:],1)
    return Znew

def Zupdate2(Xdict,Ydict,rho):
    Xmean=np.mean(np.array(list(Xdict.values())),0)
    Ymean=np.mean(np.array(list(Ydict.values())),0)
    Znew=(Xmean+Ymean/rho).copy()
    #print (V,'V')
    #Znew=np.zeros(V.shape)
    #for i in range(V.shape[0]):
        #print (V[i,:],'v')
        #Znew[i,:]=VecSimplex(V[i,:],1)
    return Znew


def ADMMall(A,B,D,rho,itr_all):
    row,column=np.shape(A)
    Xdict={}
    Ydict={}
    Za=np.zeros((row,column))
    Xdict['proj']=np.zeros((row,column))
    Ydict['proj']=np.zeros((row,column))
    Xdict['proj2']=np.zeros((row,column))
    Ydict['proj2']=np.zeros((row,column))
    for i in range(row):
        for j in range(column):
            Xdict[i,j]=np.zeros((row,column))
            Ydict[i,j]=np.zeros((row,column))
    Loss=[]
    Dual=[]
    Primal=[]
    Time=[]
    for itr in range(itr_all):
        t1=time.time()
        Z_old=Za.copy()
        for i in range(row):
            for j in range(column):
                Xdict[i,j]=Psquare(A,B,Ydict[i,j],Za,D,rho,i,j)
        Xdict['proj']=Xupdate(Za,Ydict['proj'],rho)
        Xdict['proj2']=Xupdate2(Za,Ydict['proj2'],rho)
        #Za=Zupdate(Xdict,Ydict,rho)
        Za=Zupdate2(Xdict,Ydict,rho)
        for i in range(row):
            for j in range(column):
                Ydict[i,j]=Ydict[i,j]+rho*(Xdict[i,j]-Za)
        Ydict['proj']=Ydict['proj']+rho*(Xdict['proj']-Za)
        Ydict['proj2']=Ydict['proj2']+rho*(Xdict['proj2']-Za)
        c=np.dot(A,Za)-np.dot(Za,B)
        loss=np.sum(c*c)+np.trace(np.dot(Za.T,D))
        Loss.append(loss)
        dual=np.linalg.norm(Za-Z_old)
        primal=0
        for key in Xdict:
            primal+=(np.linalg.norm(Xdict[key]-Za))**2
        Dual.append(dual)
        Primal.append(primal)
        t2=time.time()
        Time.append(t2-t1)
        #Za=Za_new.copy()
        #for key in Xdict:
            #print (np.linalg.norm(Xdict[key]-Za),'itr', itr)
    return Za,Loss,Dual,Primal,Time
