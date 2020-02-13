

import numpy as np
import networkx as nx
import tensorflow as tf
import math
import random
import time
import node2vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from keras.utils import np_utils
import sys
import pickle


def FactorGraph(Amatrix, embedding, lam, epsilon, inner_iteration_num):
    G=nx.from_numpy_matrix(Amatrix)	
    G = G.to_undirected()
    Alist=list(G.edges())
    Value=True
    num=0
    while (Value):
        random.shuffle(Alist)
        ddt=1
        embednew=embedding.copy()
        for (ki,kj) in Alist:
            eta=1./np.sqrt(ddt)
            z_ij=np.dot(embednew[ki,:],embednew[kj,:])
            Pij=Amatrix[ki,kj]-z_ij
            Ei=embednew[ki,:]+eta*(Pij*embednew[kj,:]-lam*embednew[ki,:])
            Ej=embednew[kj,:]+eta*(Pij*embednew[ki,:]-lam*embednew[kj,:])
            embednew[ki,:]=Ei
            embednew[kj,:]=Ej
            ddt+=1
        if (np.linalg.norm(embednew-embedding)**2)<=epsilon:
            Value=False
        else:
            embedding=embednew.copy()
        num+=1
        if num>=inner_iteration_num:
            Value=False
        else:
            pass
    return embedding
