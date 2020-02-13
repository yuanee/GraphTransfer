#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" 
Created on Tue Mar 26 16:28:49 2019

@author: Andrey Gritsenko
         SPIRAL Group
         Electrical & Computer Engineering
         Northeastern University
"""

"""
Module loads data and also creates synthetic data
"""

import numpy as np
import GraphProcessing as gp


def createSynthetic(save_path, n_nodes, n_labels, features, labels, B_from_A):
    """Creates two new synthetic graphs. The second graph is a permutation of the first graph
    
    INPUTS:
        SAVE_PATH Directory path, where new synthetic data will be saved
        N_NODES Number of nodes in each of new synthetic graphs
        N_LABELS Number of labels. Each node is assigned to a single label
        FEATURES Type of node features
        LABELS Type of node labels
        SWAP_EDGES Regulates whether the second graph will have some new edges and will be missing some of the old ones, 
            compared to the first graph
    
    OUTPUTS:
        New synthetic data
    """
    
    if n_labels == 4:
        A, Alabels = gp.FourClassGraph(n_nodes, labels=labels)
    else: # n_labels == 6
        A, Alabels = gp.SixClassGraph(n_nodes, labels=labels)
    B, Blabels = np.copy(A), np.copy(Alabels)
    
    ntrain = 0.8
    Aindices = np.random.permutation(len(A))
    Atrain = Aindices[:np.ceil(ntrain * len(A)).astype(int)]
    Atest = Aindices[np.ceil(ntrain * len(A)).astype(int):]
    
    # randomly permute nodes
    B, Blabels, P = gp.PermuteGraph(B, Labels=Blabels)
    # randomly add/remove edges
    if B_from_A=='modify':
        B = gp.ReverseEdges(B, .01)
    
    if features.find('index')>=0: # node feature is index of node
#        n_features = 1
        Afeatures = np.array(range(len(A))).reshape((len(A),1))
        if B_from_A=='modify':
            Bfeatures = np.matmul(P.T, np.array(range(len(B))).reshape((len(B),1)))
        else:
            Bfeatures = np.array(range(len(B))).reshape((len(B),1))
    elif features.find('adj')>=0: # node feature vector is its neighbors 
        Afeatures = np.copy(A)
        Bfeatures = np.copy(B)
#        n_features = A.shape[1]
    elif features.find('neighbor')>=0: # node feature vector is size of its neighborhood 
        Afeatures = gp.NeighborhoodSize(A)
        Bfeatures = gp.NeighborhoodSize(B)
#        n_features = Afeatures.shape[1]
    elif features.find('heat')>=0: # node feature vector is heat kernel signature of node
        Afeatures = gp.HeatKernelSignature(A)
        Bfeatures = gp.HeatKernelSignature(B)
#        n_features = Afeatures.shape[1]
    else: # node feature vector is one-hot coding vector
        features = 'onehot'
        Afeatures = np.eye(len(A))
        Bfeatures = np.eye(len(B))
    n_features = Afeatures.shape[1]
    
    fmt_features = '%.4f'
    if features.find('index')>=0:
        fmt_features = '%d'
    
    fmt_labels = '%d'
    if labels=='pagerank' or labels=='infection':
        fmt_labels = '%.6f'
    
    np.savetxt(save_path + 'GraphA.txt', A, fmt='%d')
    np.savetxt(save_path + 'GraphATrain.txt', Atrain, fmt='%d')
    np.savetxt(save_path + 'GraphATest.txt', Atest, fmt='%d')
    np.savetxt(save_path + 'GraphAFeatures_' + features + '.txt', Afeatures, fmt=fmt_features)
    np.savetxt(save_path + 'GraphALabels_' + labels + '.txt', Alabels, fmt=fmt_labels)
    
    np.savetxt(save_path + 'GraphB.txt', B, fmt='%d')
    np.savetxt(save_path + 'GraphBFeatures_' + features + '.txt', Bfeatures, fmt=fmt_features)
    np.savetxt(save_path + 'GraphBLabels_' + labels + '.txt', Blabels, fmt=fmt_labels)
    
    np.savetxt(save_path + 'ABCorrespondence.txt', P, fmt='%d')
    
    return n_nodes, n_features, n_labels, A, Afeatures, Alabels, Atrain, Atest, B, Bfeatures, Blabels

def loadSynthetic(load_path, features, labels, transmode, visualize=False):
    """
    Loads synthetic data
    
    INPUTS:
        LOAD_PATH Path the data directory
        FEATUES Specifies type of features to be used
        LABELS Specifies type of labels to be used
        TWO_GRAPHS Specifies whether two graphs should be loaded
        TRANSFER_LEARNING Specifies if permutation matrix P should be created
        VISUALIZE Specifies whether graphs should be plotted
        SAVE_PATH Path where graph plots should be saved
    
    OUTPUTS:
        NNODES Number of nodes in graphs
        NFEATURES Number of node features
        NLABELS Numbeer of node labels
        A Adjacency matrix of the first graph
        AFEATURES Node features of the first graph
        ALABELS Node labels of the first graph
        ATRAIN Indices of nodes from the first graph in the train set
        ATEST Indices of nodes from the second graph in the test set
        B Adjacency matrix of the second graph, if it is used 
        BFEATURES Node features of the second graph, if it is used
        BLABELS Node labels of the second graph, if it is used
    """
#    load_path = params.load_path + 'synthetic/'
#    features = params.features.lower()
#    labels = params.labels.lower()
    
    A = np.loadtxt(load_path + 'GraphA.txt').astype(int)
    try:
        Atrain = np.loadtxt(load_path + 'GraphATrain.txt').astype(int)
        Atest = np.loadtxt(load_path + 'GraphATest.txt').astype(int)
    except:
        ntrain = 0.8
        Aindices = np.random.permutation(len(A))
        Atrain = Aindices[:np.ceil(ntrain * len(A)).astype(int)]
        Atest = Aindices[np.ceil(ntrain * len(A)).astype(int):]
    
    Afeatures = np.loadtxt(load_path + 'GraphAFeatures_' + features + '.txt')
    nnodes = A.shape[0]
    if Afeatures.ndim == 1:
         Afeatures = Afeatures.reshape(-1,1)
    nfeatures = Afeatures.shape[1]
    
    Alabels = np.loadtxt(load_path + 'GraphALabels_' + labels + '.txt')
    if Alabels.ndim == 1:
         Alabels = Alabels.reshape(-1,1)
    nlabels = Alabels.shape[1]
    
    if transmode == '1graph':
        B, Bfeatures, Blabels = None, None, None
    else:
        B = np.loadtxt(load_path + 'GraphB.txt').astype(int)
        Bfeatures = np.loadtxt(load_path + 'GraphBFeatures_' + features + '.txt')
        if Bfeatures.ndim == 1:
            Bfeatures = Bfeatures.reshape(-1,1)
        Blabels = np.loadtxt(load_path + 'GraphBLabels_' + labels + '.txt')
        if Blabels.ndim == 1:
            Blabels = Blabels.reshape(-1,1)
    
#    if params.visualize:
    if visualize:
        from SaveData import plotGraph
        plotGraph(A, Alabels, file_name="GraphA_"+str(nnodes)+"_"+str(nlabels), save_path=load_path)
        if B is not None:
            plotGraph(B, Blabels, file_name="GraphB_"+str(nnodes)+"_"+str(nlabels), save_path=load_path)
    
    return nnodes, nfeatures, nlabels, A, Afeatures, Alabels, Atrain, Atest, B, Bfeatures, Blabels

def loadRealWorld(load_path, transmode, labels, B_from_A):
    suffix = '_split' if B_from_A == 'split' else ''
    A = np.loadtxt(load_path + 'GraphA' + suffix + '.txt').astype(int)
    nnodes = A.shape[0]
    Afeatures = np.array(range(len(A))).reshape((len(A),1))
    nfeatures = Afeatures.shape[1]
    Alabels = np.loadtxt(load_path + 'GraphALabels_' + labels + suffix + '.txt')
    if Alabels.ndim == 1:
         Alabels = Alabels.reshape(-1,1)
    nlabels = Alabels.shape[1]
    
    if labels == 'cluster':
        Atrain = np.loadtxt(load_path + 'GraphATrain.txt').astype(int)
        Atest = np.loadtxt(load_path + 'GraphATest.txt').astype(int)
    else:
        ntrain = 0.8
        Aindices = np.random.permutation(len(A))
        Atrain = Aindices[:np.ceil(ntrain * len(A)).astype(int)]
        Atest = Aindices[np.ceil(ntrain * len(A)).astype(int):]
    
    if transmode == '1graph':
        B, Bfeatures, Blabels = None, None, None
    else:
        if B_from_A == 'permute':
            B = np.loadtxt(load_path + 'GraphB.txt').astype(int)
        elif B_from_A == 'modify':
            B = np.loadtxt(load_path + 'GraphB_modify.txt').astype(int)
        else:
            B = np.loadtxt(load_path + 'GraphB' + suffix + '.txt').astype(int)
        Bfeatures = np.array(range(len(B))).reshape((len(B),1))
        Blabels = np.loadtxt(load_path + 'GraphBLabels_' + labels + suffix + '.txt')
        if Blabels.ndim == 1:
            Blabels = Blabels.reshape(-1,1)
#    if 'email' in load_path:
#            return 936, 21426, 28, None, None, None, None, None, None, None, None
#    elif 'disease' in load_path:
#        return 0, 0, 0, None, None, None, None, None, None, None, None
#    elif 'facebook' in load_path:
#        return 0, 0, 0, None, None, None, None, None, None, None, None
#    elif 'zachary' in load_path:
#        return 0, 0, 0, None, None, None, None, None, None, None, None
    return nnodes, nfeatures, nlabels, A, Afeatures, Alabels, Atrain, Atest, B, Bfeatures, Blabels

def loadData(load_path, features, labels, transmode, B_from_A='permute', visualize=False):
    """Master function to load dataset
    
    INPUTS:
        LOAD_PATH Path the data directory
        DATASET Dataset name to be loaded
        FEATUES Specifies type of features to be used
        LABELS Specifies type of labels to be used
        TWO_GRAPHS Specifies whether two graphs should be loaded
        TRANSFER_LEARNING Specifies if permutation matrix P should be created
        VISUALIZE Specifies whether graphs should be plotted
        SAVE_PATH Path where graph plots should be saved
    
    OUTPUTS:
        NNODES Number of nodes in graphs
        NFEATURES Number of node features
        NLABELS Numbeer of node labels
        A Adjacency matrix of the first graph
        AFEATURES Node features of the first graph
        ALABELS Node labels of the first graph
        ATRAIN Indices of nodes from the first graph in the train set
        ATEST Indices of nodes from the second graph in the test set
        B Adjacency matrix of the second graph, if it is used 
        BFEATURES Node features of the second graph, if it is used
        BLABELS Node labels of the second graph, if it is used
        P Permutation matrix from graph A to graph B, if it is used
    """    
    if 'synthetic' in load_path:
        try:
            return loadSynthetic(load_path, features, labels, transmode, visualize)
        except:
            n_labels = int(labels[-1])
            n_nodes = 100 if n_labels==4 else 120
            return createSynthetic(load_path, n_nodes, n_labels, features, labels, B_from_A)
    else:
        try:
            return loadRealWorld(load_path, transmode, labels, B_from_A)
        except:
            print("Cannot load data: wrong dataset name.")
            return 0, 0, 0, None, None, None, None, None, None, None, None
    







