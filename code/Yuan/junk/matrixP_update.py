#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 22:42:40 2019

@author: yuanneu
"""


import numpy as np
import random

from cvxopt import matrix
import numpy as np
from cvxopt import solvers


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

def VecOpt(v,r):
    """
    max <s,v>
    subject to <s,1>=r
    """
    d=len(v)
    y=v-(np.sum(v)-r)/d*np.ones(d)
    return y


def ADMMProject(V,rho):
    """
    the ADMM algorithm to solve:
    min ||S-V||_{2}^{2}
    subject to S>=0, S1=1, 1^{T}S=1^{T}
    where V=P^(k)-gamma*∇P^(k)
    """
    size=np.shape(V)[0]
    Z=V.copy()
    Q=np.zeros((size,size))
    M=np.zeros((size,size))
    Y1=np.zeros((size,size))
    Y2=np.zeros((size,size))
    Value=True
    ddt=0
    while (Value==True):
        A1=(V+rho/2.0*Z-1/2.*Y1)/(1+rho/2.)
        A2=(rho*Z-Y2)/rho
        for i in range(size):
            Q[i,:]=VecSimplex(A1[i,:],1)
            M[:,i]=VecOpt(A2[:,i],1)
        Z=1/2.*(Q+M+1/rho*(Y1+Y2))
        Y1+=rho*(Q-Z)
        Y2+=rho*(M-Z)
        ddt+=1
        qz_norm=np.linalg.norm(Q-Z)
        mz_norm=np.linalg.norm(M-Z)
        if (qz_norm<=1e-5) and (mz_norm<=1e-5):
            Value=False
        else:
            pass
    return Z



def ADMMProjectDot(d_P,rho):
    """
    the ADMM algorithm to solve:
    min <S, V>
    subject to S>=0, S1=1, 1^{T}S=1^{T}
    V=∇||AP-PB||_{2}^{2}+D
    """
    
    size=np.shape(d_P)[0]
    Z=d_P.copy()
    S=np.zeros((size,size))
    M=np.zeros((size,size))
    Y1=np.zeros((size,size))
    Y2=np.zeros((size,size))
    Value=True
    while (Value==True):
        A1=Z-(Y1+d_P)/rho
        A2=(rho*Z-Y2)/rho
        for i in range(size):
            S[i,:]=VecSimplex(A1[i,:],1)
            M[:,i]=VecOpt(A2[:,i],1)
        Z=(S+M+1/rho*(Y1+Y2))/2.
        Y1+=rho*(S-Z)
        Y2+=rho*(M-Z)
        qz_norm=np.linalg.norm(S-Z)
        mz_norm=np.linalg.norm(M-Z)
        if (qz_norm<=1e-5) and (mz_norm<=1e-5):
            Value=False
    return Z


def descent_fun(A,B,P,D):
    """
    the gradient descent for ||AP-PB||_{2}^{2}
    """
    C=np.dot(A,P)-np.dot(P,B)   
    Y=np.dot(A.T,C)-np.dot(C,B.T)+D
    return Y



def ProxmialOpt(A,B,D,gamma,rho,itr_proj):
    """
    The Proximal Gradient Descent algorithm main structure
    Solve the problem:
    min ||AP-PB||_{2}^{2}+tr(P^{T}D)
    subject to: P>=0, P1=1, 1^{T}P=1^{T}.
    """
    size=np.shape(A)[0]
    Pnew=np.identity(size)
    for itr in range(itr_proj):
        Y=descent_fun(A,B,Pnew,D)
        Proj=Pnew-gamma*Y
        Pnew=ADMMProject(Proj,rho)
    return Pnew

def Frank_Wolfe(A,B,D,rho,itr_proj):
    """
    The Frank-Wofle algorithm main structure
    Solve the problem:
    min ||AP-PB||_{2}^{2}+tr(P^{T}D)
    subject to: P>=0, P1=1, 1^{T}P=1^{T}.
    """
    size=np.shape(A)[0]
    P=np.identity(size)
    for k in range(itr_proj):
        nabla_P=descent_fun(A,B,P,D)
        S_optimal=ADMMProjectDot(nabla_P,rho)
        Pnew=P+2./(k+2)*(S_optimal-P)
        P=Pnew.copy()
    return P

def rewrite(A,B,D):
    """
    rewrite ||AP-PB||_{2}^{2}+tr(P^{T}D) in a quadratic form with respect to P element-wisely.
    """
    size=np.shape(A)[0]
    P1=np.zeros((size,size))
    P2=np.zeros((size,size))
    Quadratic=np.zeros((size*size,size*size))
    for i in range(size):
        for j in range(size):
            l=i*size+j
            P1[i,j]=1
            C1=np.dot(A,P1)-np.dot(P1,B)
            for k in range(size):
                for t in range(size):
                    m=k*size+t
                    P2[k,t]=1
                    C2=np.dot(A,P2)-np.dot(P2,B)
                    Quadratic[l,m]=np.trace(np.dot(C1,C2.T))
                    P2[k,t]=0
            P1[i,j]=0  
    Q=2*Quadratic
    p= D.reshape(1,-1)[0,:]   
    G = -np.identity(size*size)
    h = np.zeros(size*size)
    A1= np.zeros((2*size,size*size))
    for i in range(size):
        A1[i,range(size*i,size*i+size)]=np.ones(size)
        for j in range(size):
            A1[size+i,size*j+i]=1
    b= np.ones(2*size)
    G=np.concatenate((G,A1,-A1),0)
    h=np.concatenate((h,b,-b))
    return Q,p,G,h


def cvx_P(A,B,D):
    """
    solve min ||AP-PB||_{2}^{2}+tr(P^{T}D) using cvxopt
    """
    size=np.shape(A)[0]
    Q,p,G,h=rewrite(A,B,D)
    Q=matrix(Q)
    p=matrix(p)
    G=matrix(G)
    h=matrix(h)
    sol=solvers.qp(Q, p, G, h)
    vec_P=np.array(sol['x'])
    Pnew=vec_P.reshape((size,size))
    return Pnew







