#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:59:29 2019

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


def Derivative(A,B,D,P,order,mode):
    dimA=A.shape[0]
    C=np.dot(A,P)-np.dot(P,B) 
    if mode=='square':       
        grad_P=2*(np.dot(A.T,C)-np.dot(C,B.T))
    else:
        Corder=np.sum(C**order)
        if Corder<=10**(-8):
            cp=0
        else:
            cp=Corder**(1./order-1)
        #cp=(np.sum(C**order))**(1./order-1)
        Cp1=C**(order-1)
        grad_P=cp*(np.dot(A.T,Cp1)-np.dot(Cp1,B.T))
    grad_P+=D
    return grad_P

def Loss_P(A,B,D,P,order,mode):
    C=np.dot(A,P)-np.dot(P,B) 
    if mode=='square':
        loss=np.sum(C**2)
    else:
        loss=(np.sum(C**order))**(1./order)
    loss+=np.sum(P*D)
    return loss


class Frank_P:
    """
    The Frank Wolfe class 
    To solve ||AP-PB||_{2}^{2}+tr(P^{T}D) using cvxopt
    """
    def __init__(self, A, B, D, P, order, norm_mode, gamma=0.1):
        self.A = A
        self.B = B
        self.D = D
        self.P = P
        self.gamma = gamma
        self.order=order
        self.norm=norm_mode
        
    def initialize(self):
        pass
        
    def first_fun(self):
        nablaP=Derivative(self.A,self.B,self.D,self.P,self.order,self.norm)
        return nablaP  
    
    def iteration(self, epsilon, itr_num):
        itr=0
        while(True):
            nabla_P=self.first_fun()
            S_optimal=V_dot(nabla_P)
            delta_P=S_optimal-self.P  
            eta=2/(itr+2)
            dual=-np.sum(nabla_P*delta_P)
            P_new=self.P+eta*delta_P
            print(np.linalg.norm(np.dot(self.A,P_new)-np.dot(P_new,self.B)))
            self.P=P_new.copy()
            #print (np.linalg.norm(np.dot(self.A,self.P)-np.dot(self.P,self.B))**2+np.sum(self.P*self.D))
            itr+=1
            if itr>=itr_num:
                break
            else:
                pass
            if dual<=epsilon:
                break
            else:
                pass
        return P_new
    
class Frank_Gradient(Frank_P):
    """
    The Frank Wolfe class 
    To solve ||P-V||_{2}^{2} 
    which V=
    """
    def initialize(self):
        de_P=Derivative(self.A,self.B,self.D,self.P,self.order,self.norm)
        #print (self.gamma)
        self.V=self.P-self.gamma*de_P
        #print (np.linalg.norm(self.V))
        
    def first_fun(self):        
        nabla_P=2*(self.P-self.V)
        return nabla_P  
      

def MapZero(Z,Y,rho,mode):
    V=Z-Y/rho
    Xtrace=V.copy()
    if mode=='column':
        for i in range(V.shape[1]):
            Xtrace[:,i]=VecSimplex(V[:,i],1)
    else:
        for i in range(V.shape[0]):
            Xtrace[i,:]=VecSimplex(V[i,:],1)
    return Xtrace


class ADMM_OBJ:
    """
    The ADMM class to solve:
    solve min ||P-V||_{2}^{2}
    P1=P^{T}1=0
    P>=0
    """
    def __init__(self,V):
        self.V=V
        self.X={}
        self.Y={}
        self.Z=np.zeros(V.shape)
    def loop(self,rho,epsilon,itr_num):
        self.Y['P']=np.zeros(self.V.shape)
        self.Y['row']=np.zeros(self.V.shape)
        self.Y['column']=np.zeros(self.V.shape)
    
        dual=10**10
        primal=10**10
        itr=0
        while((dual>=epsilon)|(primal>=epsilon)):
            #t1=time.time()
            Z_old=self.Z.copy()
            self.X['P']=(2*self.V+rho*self.Z-self.Y['P'])/(2+rho)
            self.X['row']=MapZero(self.Z,self.Y['row'],rho,'row')
            self.X['column']=MapZero(self.Z,self.Y['column'],rho,'column')
            self.Z=np.mean(np.array(list(self.X.values())),0)
            primal=0
            for key in self.Y:
                y_d=self.X[key]-self.Z
                self.Y[key]=self.Y[key]+rho*y_d
                primal+=np.linalg.norm(y_d)**2
            primal=np.sqrt(primal)
            dual=np.sqrt(3)*rho*np.linalg.norm(self.Z-Z_old) 
            itr+=1
            if itr>=itr_num:
                break
            else:
                pass
        return self.Z
    
def rewrite(V):
    """
    rewrite ||P-V||_{2}^{2} and constraint P1=P^{T}1=1, P>=0 in a quadratic form.
    """
    size=np.shape(V)[0]
    Q=2*np.identity((size*size))  
    p= -2*V.reshape(1,-1)[0,:]   
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
    

def quadratic_fun(A,B):
    """
    rewrite ||AP-PB||_{2}^{2} as a p^{T}Qp form.
    """
    size=np.shape(A)[0]
    Quadratic=np.zeros((size*size,size*size))
    for row in range(size*size):
        ki,kj=int(row/size),int(row)%int(size)
        #print (ki,kj)
        Quadratic[row,row]=np.dot(A[:,ki],A[:,ki])+np.dot(B[kj,:],B[kj,:])-2*A[ki,ki]*B[kj,kj]
    for row in range(1,size*size):
        for column in range(0,row):
            ki,kj=int(row/size),row%size
            km,kn=int(column/size),column%size
            #print ((ki,kj),(km,kn),row,column)
            if ((ki!=km)&(kj!=kn)):
                Quadratic[row,column]=-(A[ki,km]*B[kj,kn]+A[km,ki]*B[kn,kj])
            else:
                if (ki==km):
                    Quadratic[row,column]=-A[ki,ki]*(B[kj,kn]+B[kn,kj])+np.dot(B[kn,:],B[kj,:])
                else:
                    Quadratic[row,column]=-B[kj,kj]*(A[ki,km]+A[km,ki])+np.dot(A[:,ki],A[:,km])
    for row in range(0,size*size-1):
        for column in range(row+1,size*size):
            Quadratic[row,column]= Quadratic[column,row]
    return Quadratic
            
def rewriteP(A,B,D):
    """
    rewrite ||AP-PB||_{2}^{2}+tr(P^{T}D) in a quadratic form with respect to P element-wisely.
    """
    size=np.shape(A)[0]
    Quadratic=quadratic_fun(A,B)
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


def GenerateC(Am,Bm,key):
    ki,kj=key
    a_v,b_v=Am[ki,:],-Bm[:,kj]
    z=np.array([a_v,b_v])
    z[0,ki]+=b_v[kj]
    z[1,kj]=0
    c_value=np.linalg.norm(z)**2
    return [c_value, z]


def X_update(Xdict,Ydict,Cdict,D,Z,size,rho):
    for i in range(size):
        for j in range(size):
            cvalue,vector=Cdict[i,j]
            zm=np.zeros((2,D.shape[0]))
            zm[0,:]=Z[:,j]
            zm[1,:]=Z[i,:]
            zm[0,i]-=D[i,j]/rho
            V_v=zm-(Ydict[(i,j)]/rho)
            xm=V_v-np.sum(vector*V_v)/(rho/2.+cvalue)*vector
            xm[1,j]=0.0
            Xdict[(i,j)]=xm 
    Xdict['row']=MapZero(Z,Ydict['row'],rho,'row')
    Xdict['column']=MapZero(Z,Ydict['column'],rho,'column')
            


def Z_update(Xdict,Ydict,Z,size,rho):
    zm=np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            xm=Xdict[(i,j)]
            #ym=Ydict[(i,j)]
            zm[:,j]+=(xm[0,:])
            zm[i,:]+=(xm[1,:])
    zm+=(Xdict['row'])
    zm+=(Xdict['column'])
    zm=zm/(2*size+1)
    dual=np.sqrt(2*size+1)*rho*np.linalg.norm(zm-Z)
    return zm, dual

def Y_update(Xdict,Ydict,Z,rho,size): 
    primal=0
    for i in range(size):
        for j in range(size):
            zm=np.array([Z[:,j],Z[i,:]])
            y_d=Xdict[(i,j)]-zm
            primal+=(np.linalg.norm(y_d)**2-y_d[1,j]**2)
            Ydict[(i,j)]+=rho*(y_d)
    y_d=Xdict['row']-Z 
    primal+=(np.linalg.norm(y_d)**2)   
    Ydict['row']+=rho*y_d
    y_d=Xdict['column']-Z
    primal+=(np.linalg.norm(y_d)**2)
    Ydict['column']+=rho*y_d
    return np.sqrt(primal)





class ADMM_optimal:
    """
    The class to solve the problem:
    solve min ||AP-PB||_{2}^{2}+tr(P^{T}D) 
    P1=P^T1=1
    P>=0
    """  
    def __init__(self,A,B,D):
        self.A=A
        self.B=B
        self.D=D
        self.C={}
        self.X={}
        self.Y={}
        self.Z=np.zeros(A.shape)
      
    def loop(self, rho,epsilon,itr_num):
        size=self.A.shape[0]
        element_list=[(i,j) for i in range(size) for j in range(size)]
        for key in element_list:
            self.C[key]=GenerateC(self.A,self.B,key)
            self.Y[key]=np.zeros((2,size))
        self.Y['row']=np.zeros((size,size)) 
        self.Y['column']=np.zeros((size,size))
        #self.Zlist=[]
        dual=10**10
        primal=10**10
        itr=0
        while((dual>=epsilon)|(primal>=epsilon)):
            X_update(self.X,self.Y,self.C,self.D,self.Z,size,rho)
            self.Z,dual=Z_update(self.X,self.Y,self.Z,size,rho)
            primal=Y_update(self.X,self.Y,self.Z,rho,size)
            #print (dual, primal,'dual')
            itr+=1
            if itr>=itr_num:
                break
            else:
                pass
        return self.Z



def cvx_P(A,B,D):
    """
    solve min ||AP-PB||_{2}^{2}+tr(P^{T}D) using cvxopt
    """
    size=np.shape(A)[0]
    Q,p,G,h=rewriteP(A,B,D)
    Q=matrix(Q)
    p=matrix(p)
    G=matrix(G)
    h=matrix(h)
    sol=solvers.qp(Q, p, G, h)
    vec_P=np.array(sol['x'])
    Pnew=vec_P.reshape((size,size))
    return Pnew

def cvx_P_gradient(V):
    """
    solve min ||S-V||_{2}^{2} using cvxopt
    subject to S1=1, S^{T}1=1, S>=0.
    """
    Q,p,G,h=rewrite(V)
    Q=matrix(Q)
    p=matrix(p)
    G=matrix(G)
    h=matrix(h)
    sol=solvers.qp(Q, p, G, h)
    vec_P=np.array(sol['x'])
    Pnew=vec_P.reshape(V.shape)
    return Pnew  


def ADMMV(V,rho=1,epsilon=0.0001,itr_num=10000):
    obj=ADMM_OBJ(V)
    Pnew=obj.loop(rho,epsilon,itr_num)
    return Pnew

def ADMMP(A,B,D):
    """
    The problem to solve min ||AP-PB||_{2}^{2}+tr(P^{T}D)
    """
    obj=ADMM_optimal(A,B,D)
    pnew=obj.loop(1,0.001,20000)
    return pnew

def Frank_optimal(A,B,D,norm=2,mode='norm'):
    """
    Frank-Wolfe method to solve:
    ||AP-PB||_{2}^{2}+tr(P^{T}D)
    P1=P^T1=1
    P>=0
    """
    P=np.identity(A.shape[0])
    obj=Frank_P(A,B,D,P,norm,mode)
    pnew=obj.iteration(0.001,2000)
    return pnew

def Frank_update(A,B,D,P,gam,norm=2,mode='norm'):
    """
    Frank-Wolfe method to solve:
    ||AP-PB||_{p}+tr(P^{T}D)
    P1=P^T1=1
    P>=0
    mode can be 'norm' or 'square'
    """
    obj=Frank_Gradient(A, B, D, P, norm, mode,gamma=gam)
    obj.initialize()
    pnew=obj.iteration(0.001,100)
    return pnew

def ADMM_update(A,B,D,P,gam,norm=2,mode='norm'):
    """
    projection of the ||AP-PB||_{p}+tr(P^{T}D)
    with gamma
    """
    dep=Derivative(A,B,D,P,norm,mode)
    V=P-gam*dep
    pnew=ADMMV(V)
    return pnew

def Cvx_update(A,B,D,P,gam,norm=2,mode='norm'):
    """
    CVXOPT to solve the problem of
    projection of the ||AP-PB||_{p}+tr(P^{T}D)
    with gamma
    """
    dep=Derivative(A,B,D,P,norm,mode)
    V=P-gam*dep
    pnew=cvx_P_gradient(V)
    return pnew




















            




