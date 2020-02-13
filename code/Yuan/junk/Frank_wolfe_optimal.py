#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:37:23 2019

@author: yuanneu
"""


import numpy as np
import math
import random
from cvxopt import matrix
from cvxopt import solvers
import time
import pickle
from scipy.optimize import linear_sum_assignment


def descent_square(A,B,D,P):
    """
    the gradient descent for ||AP-PB||_{2}^{2}+tr(P^{T}D)
    """
    C=np.dot(A,P)-np.dot(P,B)   
    Y=2*(np.dot(A.T,C)-np.dot(C,B.T))+D
    return Y

def descent_norm(A,B,D,P):
    """
    the gradient descent for ||AP-PB||_{2}+tr(P^{T}D)
    """
    C=np.dot(A,P)-np.dot(P,B)
    Y=(np.dot(A.T,C)-np.dot(C,B.T))/np.linalg.norm(C)+D
    return Y

    
def trace_eta(A,B,P,delta_P,eta):
    """
    calculate tr[(P+eta*delta_P)(P+eta*delta_P)^{T}]
    """
    z=np.dot(A,P)-np.dot(P,B)
    delta=np.dot(A,delta_P)-np.dot(delta_P,B)
    a=np.trace(np.dot(z,z.T))
    b=np.trace(np.dot(z,delta.T))
    c=np.trace(np.dot(delta,delta.T))
    square=a+2*b*eta+c*eta**2
    return square

def trace_value(A,B,D,P,delta_P):
    """
    calculate tr[(P+eta*delta_P)(P+eta*delta_P)^{T}]
    """
    z=np.dot(A,P)-np.dot(P,B)
    delta=np.dot(A,delta_P)-np.dot(delta_P,B)
    a=np.trace(np.dot(z,z.T))
    b=np.trace(np.dot(z,delta.T))
    c=np.trace(np.dot(delta,delta.T))
    d=np.trace(np.dot(delta_P.T,D))
    return a,b,c,d

def search_fun(fun,eps_search):
    Num=int(1./eps_search)
    xlist=[eps_search*i for i in range(Num+1)]    
    ylist=[fun(x) for x in xlist]    
    step_size = xlist[np.argmin(ylist)]
    return step_size

def square_linesearch(A,B,D,P,delta_P):
    a,b,c,d=trace_value(A,B,D,P,delta_P)
    gain_fun=lambda eta: (a+2*b*eta+c*eta**2)+eta*d
    step_size=search_fun(gain_fun,0.0001)
    return step_size

def norm_linesearch(A,B,D,P,delta_P):
    a,b,c,d=trace_value(A,B,D,P,delta_P)
    gain_fun=lambda eta: np.sqrt(a+2*b*eta+c*eta**2)+eta*d
    step_size=search_fun(gain_fun,0.0001)
    return step_size


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

def norm_loss_fun(a,b,d,p):
    """
    ||AP-PB||_{2}+tr(P^{T}D)
    """
    c=np.dot(a,p)-np.dot(p,b)
    loss=np.linalg.norm(c)+np.trace(np.dot(p.T,d))
    return loss

def square_loss_fun(a,b,d,p):
    """
    ||AP-PB||_{2}^{2}+tr(P^{T}D)
    """
    c=np.dot(a,p)-np.dot(p,b)
    loss=(np.linalg.norm(c))**2+np.trace(np.dot(p.T,d))
    return loss

class frank_wolfe:
    def __init__(self,A,B,D):
        self.A=A
        self.B=B
        self.D=D
        
    def loop(self,grad_fun,loss_fun,step_fun,epsilon,itr_num):
        self.loss=[]
        self.time=[]
        loss=10**10
        loss_gain=10**5
        P=np.identity(self.A.shape[0])
        itr=0
        while(loss_gain>epsilon):
            t1=time.time()
            loss_old=loss
            nabla_P=grad_fun(self.A,self.B,self.D,P)
            S_optimal=V_dot(nabla_P)
            delta_P=S_optimal-P  
#            tp1=time.time()
            eta=step_fun(self.A,self.B,self.D,P,delta_P)
#            tp2=time.time()
#            print ('time',tp2-tp1)
#            print ('eta',eta)
            P_new=P+eta*delta_P
            loss=loss_fun(self.A,self.B,self.D,P_new)
#            print (itr,loss)
            loss_gain=loss_old-loss
            self.loss.append(loss)
            P=P_new.copy()
            t2=time.time()
            self.time.append(t2-t1)
            itr+=1
            if itr>=itr_num:
                break
            else:
                pass
        self.optimal=P_new.copy()

def Non_Gradient_P(A,B,D,mode):
    """
    if mode is 'norm'
    return the optimal solutin for 
    min ||AP-PB||_{2}+tr(P^{T}D)
    subject to: P>=0, P1=1, 1^{T}P=1^{T}.
    else return the optimal solution 
    min ||AP-PB||_{2}^{2}+tr(P^{T}D)
    subject to: P>=0, P1=1, 1^{T}P=1^{T}.
    """
    non_grad_obj=frank_wolfe(A,B,D)
    if mode=='norm':
        non_grad_obj.loop(descent_norm,norm_loss_fun,norm_linesearch,0.001,200)
    else:
        non_grad_obj.loop(descent_square,square_loss_fun,square_linesearch,0.0001,400)   
    return non_grad_obj.optimal  
    

