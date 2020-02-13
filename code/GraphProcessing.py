#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 09:12:54 2019

@author: Andrey Gritsenko
         SPIRAL Group
         Electrical & Computer Engineering
         Northeastern University
"""

"""
Module contains different functions to operate on graphs
"""

import numpy as np
from scipy.sparse import csgraph
import math
from itertools import permutations


def SixClassGraph(nnodes=120, visualize=False, labels='cluster'):
    """
    Create a graph of NNODES nodes distributed in 6 equal sized classes.
    The graph is characterized by dense connections between nodes of the same class, and shallow connections between nodes of othe classes
    
    Inputs:
        NNODES Number of nodes in the graph
        
    Ouputs:
        A Adjacency matrix of the graph
        LABELS Matrix of labels (NNODES x NLABELS), where each row represents a one-hot encoding of node's label
    """
    
    A = np.zeros((nnodes,nnodes))
    nlabels = 6
    
    #Erdos-Renyi probabilities for clusters
    pd = [.75,.65,.85,.55,.75,.6]
    ps = .10

    class_size = int(np.ceil(float(nnodes)/nlabels))
    for i in range(nlabels):
        A[i*class_size:min((i+1)*class_size,nnodes), i*class_size:min((i+1)*class_size,nnodes)] = (np.random.rand(min((i+1)*class_size,nnodes)-i*class_size, min((i+1)*class_size,nnodes)-i*class_size) <= pd[i]).astype(int)
    
    shallow_connections = [[0,1],[1,2],[2,3],[2,4],[3,4],[4,5]]
    
    for i in range(len(shallow_connections)):
        x,y = shallow_connections[i]
        A[x*class_size:min((x+1)*class_size,nnodes), y*class_size:min((y+1)*class_size,nnodes)] = (np.random.rand(min((x+1)*class_size,nnodes)-x*class_size, min((y+1)*class_size,nnodes)-y*class_size) <= ps).astype(int)
    
    A = (np.triu(A,1)+np.triu(A,1).T).astype(int) 
    
    if labels=='infection' :
        Labels = getDiffusionModelLabels(A)
    elif labels=='pagerank' :
        Labels = getPagerankLabels(A)
        pass
    else: # 'cluster'
        label_list = np.eye(nlabels)
        Labels = np.array([[label_list[i,:].tolist()]*class_size for i in range(nlabels)], dtype=int).reshape(nlabels*class_size,nlabels)[:nnodes,:]
    
    if visualize: 
        from SaveData import plotGraph
        plotGraph(A, Labels)
    return A, Labels


def FourClassGraph(nnodes=100, visualize=False, labels='cluster'):
    """
    Create a graph of NNODES nodes distributed in 4 equal sized classes with guitar topology.
    The graph is characterized by dense connections between nodes of the same class, 
    shallow mutual connections between nodes of 3 classes forming a triange, and 
    shallow connections between nodes of the forth class and one of the three other classes
    
    Inputs:
        NNODES Number of nodes in the graph
        
    Ouputs:
        A Adjacency matrix of the graph
        LABELS Matrix of labels (NNODES x NLABELS), where each row represents a one-hot encoding of node's label
    """
    
    A = np.zeros((nnodes,nnodes))
    nlabels = 4
    
    #Erdos-Renyi probabilities for clusters
    pd = [.75,.45,.65,.6]
    ps = .10

    class_size = int(np.ceil(float(nnodes)/nlabels))
    for i in range(nlabels):
        A[i*class_size:min((i+1)*class_size,nnodes), i*class_size:min((i+1)*class_size,nnodes)] = (np.random.rand(min((i+1)*class_size,nnodes)-i*class_size, min((i+1)*class_size,nnodes)-i*class_size) <= pd[i]).astype(int)
    
    shallow_connections = [[0,1],[0,2],[1,2],[2,3]]
    
    for i in range(len(shallow_connections)):
        x,y = shallow_connections[i]
        A[x*class_size:min((x+1)*class_size,nnodes), y*class_size:min((y+1)*class_size,nnodes)] = (np.random.rand(min((x+1)*class_size,nnodes)-x*class_size, min((y+1)*class_size,nnodes)-y*class_size) <= ps).astype(int)
    
    A = (np.triu(A,1)+np.triu(A,1).T).astype(int) 
    
    if labels=='infection' :
        Labels = getDiffusionModelLabels(A)
    elif labels=='pagerank' :
        Labels = getPagerankLabels(A)
        pass
    else: # 'cluster'
        label_list = np.eye(nlabels)
        Labels = np.array([[label_list[i,:].tolist()]*class_size for i in range(nlabels)], dtype=int).reshape(nlabels*class_size,nlabels)[:nnodes,:]
    
    if visualize: 
        from SaveData import plotGraph
        plotGraph(A, Labels)
    return A, Labels


def PermuteGraph(Adj, Labels=None, P=None):
    """
    Permute graph according to permutation matrix P
    
    Inputs:
        ADJ Adjacency matrix of a graph to be permuted
        P Permutation matrix, such that B = P^T * A * P, where A is an old adjacency matrix and B is a new one
        LABELS List of node labels
        
    Outputs:
        BSIM Adjacency matrix of a new graph after permutation
        BLABELS List of node labels after permutation
    """
    
    if P is None:
        P = np.eye(len(Adj),dtype=int)[np.random.permutation(len(Adj))]
        
    new_Adj = np.matmul(np.matmul(P.T, Adj), P)
    
    new_Labels = np.matmul(P.T, Labels) if Labels is not None else None
    return new_Adj, new_Labels, P


def ReverseEdges(A, Prob=0.05):
    """
    Invert elements of graph's adjacency matrix with probability Prob
    
    Inputs:
        A Adjacency matrix of a graph to be permuted
        PROB Probability of inverting the adjacency relationship between graph nodes
        
    Outputs:
        B Adjacency matrix of a new graph after edge inversion
    """
    
    if Prob is None or Prob<0 or Prob>=1:
        Prob = 0.05
    
    triup=[]
    for i in range(len(A)):
        triup.extend((np.array(range(len(A)))+i*len(A))[i+1:])
    mask = np.random.choice(triup, int(np.ceil(Prob * len(triup))))
    
    B1=A.reshape((-1,1)).copy()
    B1[mask] = ~B1[mask]+2 # inverts elements of similarity matrix
    
    B = np.triu(B1.reshape(A.shape),1) + np.triu(B1.reshape(A.shape),1).T # preserves simmetry of new adjacency matrix 
    return B


def SplitGraph(A):
    """
    Splits a graph in two with the sparsest cut (Fiedler vector). The objective function favors solutions that are both sparse (few edges crossing the cut) and balanced (close to a bisection)
    Due to the specifics of the proposed framework, we split graph in two equal subgraphs
    """
    import networkx as nx
    import statistics as stat
    
    F = nx.linalg.algebraicconnectivity.fiedler_vector(nx.from_numpy_matrix(A))
    f = stat.median(F)
    Aind = F>f
    Bind = F<f
    
    A1 = (A[Aind,:])[:,Aind]
    B1 = (A[Bind,:])[:,Bind]
    
    return A1, B1, F, f


def getPagerankLabels(adjacency, iterations=100, tolerance=1e-4):
    
    import networkx as nx
    
    G = nx.from_numpy_matrix(adjacency.copy())
    pr = nx.pagerank(G, max_iter=iterations if iterations is not None else 100, tol=tolerance if tolerance is not None else 1e-4)
    PR = np.array(list(pr.values())).reshape(-1,1)
    
#    from sklearn import preprocessing
#    PR = preprocessing.normalize(np.array(list(pr.values())).reshape(1,-1)).T # normalized pagerank
    
#    import matplotlib.pyplot as plt
#    plt.imshow(np.repeat(PR.reshape((len(PR),1)),10,axis=1))
    return PR


def getDiffusionModelLabels(adjacency, repetitions=1000, threshold=0.5, patient0=None):
    
    import networkx as nx
    import ndlib.models.ModelConfig as mc
    from ndlib.models.epidemics.IndependentCascadesModel import IndependentCascadesModel
    
    if patient0 is None:
        patient0 = nx.center(nx.from_numpy_matrix(adjacency))[0] # A-4: 50, B-4: 71; A-6: 42, B-4: 26

    A = adjacency.copy()
    g = nx.from_numpy_matrix(A)
    cfg = mc.Configuration()
    cfg.add_model_initial_configuration("Infected", set(list([patient0])))
    
    if threshold is None:
        threshold = 0.5
    for e in g.edges():
        cfg.add_edge_configuration("threshold", e, threshold)
        
    model = IndependentCascadesModel(g)
    model.set_initial_status(cfg)
    
    if repetitions is None:
        repetitions = 1000
    res_probs = np.zeros((A.shape[0],repetitions))
    res_Probs = np.zeros(res_probs.shape)
    
    iteration_results = []
    for rep in range(repetitions):
        model.reset()
        itr = 0
        while True:
            itr += 1
            iteration_result = model.iteration()
            iteration_results.append(iteration_result)
            
            for node,status in iteration_result['status'].items():
                if status==1:
                    res_probs[node,rep] = itr
                    
            if len(iteration_result['status']) == 0:
                break
            
        res_Probs[res_probs[:,rep]>1,rep] = 1.0 / (res_probs[res_probs[:,rep]>1,rep] - 1.)
    
    res_Prob = np.mean(res_Probs,1)
    res_Prob[patient0] = np.mean(res_Prob[np.argwhere(adjacency[patient0,:])][:,0])
    
#    import matplotlib.pyplot as plt
#    plt.imshow(np.repeat(res_Prob.reshape((len(res_Prob),1)),10,axis=1))
    return res_Prob


def HeatKernelSignature(adjacency_matrix, ntime_stamps=10):
    """
    Compute node features as heat diffusion on graph
    """
#    adjacency_matrix = np.array([[0,1,1,1,0,0,0,0,0,0,0,0],
#                                  [1,0,1,1,0,0,0,0,0,0,0,0],
#                                  [1,1,0,1,0,0,0,0,0,0,0,0],
#                                  [1,1,1,0,1,1,0,0,0,0,0,0],
#                                  [0,0,0,1,0,1,0,1,0,0,0,0],
#                                  [0,0,0,1,1,0,1,0,0,1,1,1],
#                                  [0,0,0,0,0,1,0,1,1,0,0,0],
#                                  [0,0,0,0,1,0,1,0,1,0,0,0],
#                                  [0,0,0,0,0,0,1,1,0,0,0,0],
#                                  [0,0,0,0,0,1,0,0,0,0,1,1],
#                                  [0,0,0,0,0,1,0,0,0,1,0,1],
#                                  [0,0,0,0,0,1,0,0,0,1,1,0]])
    
    HKS = np.zeros((len(adjacency_matrix),ntime_stamps))
    
    graph_laplacian = csgraph.laplacian(adjacency_matrix)
    
    [eigval, eigvec] = np.linalg.eigh(graph_laplacian)
    
    time = np.logspace(math.log10(.5/eigval[-1]), math.log10(1/eigval[1]), num=ntime_stamps)
    for t in range(ntime_stamps):
        time
        HKS[:,t] = np.diag(np.matmul(eigvec, np.matmul(np.diagflat(np.exp(-time[t]*eigval)), eigvec.T)))
    return HKS


def NeighborhoodSize(adj_mat, dist=None):
    """
    Compute node features as size of node neighborhood
    """
    import networkx as nx
    from sklearn import preprocessing
    
    G = nx.from_numpy_matrix(adj_mat)
    
    if dist == None:
        dist = int(np.floor(nx.diameter(G)*.75))
    
    features = np.zeros((G.order(),dist))
    
    for node in range(G.order()):
        for step in range(dist):
            features[node,step] = nx.ego_graph(G, node, radius=step+1, center=False, undirected=True).order()
    
#    features_n = preprocessing.normalize(features) # scale input to unit norm
    features_s = preprocessing.scale(features) # scale input to zero-mean and unit-variance
#    features_ns = preprocessing.scale(features_n)
#    features_sn = preprocessing.normalize(features_s)
    return features_s


def GaussianKernelSimilarity(graph_representation, u, v, sigma=1):
    """
    Implements gaussian kernel similarity
    
    Inputs:
        GRAPH_REPRESENTATION Representation of graph in matrix form, where each row represents a node
        U, V Indeces of graph nodes pair, between which similarity is computed
        SIGMA Free parameter that controls smoothness of kernel
        
    Outputs:
        Measure of similarity between two nodes
    """
    # TODO : rewrite using graph laplacian
    return np.exp(-np.sum((graph_representation[u,:]-graph_representation[v,:])**2) / (2*sigma**2))


def HeatKernelSimilarity(graph_representation, u, v, time=1):
    """
    Implements heat kernel similarity
    
    Inputs:
        GRAPH_REPRESENTATION Representation of graph in matrix form, where each row represents a node
        U, V Indeces of graph nodes pair, between which similarity is computed
        TIME Time parameter in the equation of heat diffusion
        
    Outputs:
        Measure of similarity between two nodes
    """
    # TODO : rewrite using graph laplacian
    return np.exp(-np.sum((graph_representation[u,:]-graph_representation[v,:])**2) / (4*time)) / (4*np.pi*time)**(float(len(graph_representation[u,:]/2)))


def InnerProductSimilarity(graph_representation, u, v):
    """
    Implements inner product similarity
    
    Inputs:
        GRAPH_REPRESENTATION Representation of graph in matrix form, where each row represents a node
        U, V Indeces of graph nodes pair, between which similarity is computed
        
    Outputs:
        Measure of similarity between two nodes
    """
    return 2-np.inner(graph_representation[u,:], graph_representation[v,:])


def NodeSimilarity(A,i,j,mode='adjacency'):
    """
    Compute similarity between two nodes
    """
    if mode=='gauss':
        return GaussianKernelSimilarity(A,i,j)
    elif mode=='heat':
        return HeatKernelSimilarity(A,i,j)
    elif mode=='innerproduct':
        return InnerProductSimilarity(A,i,j)
    else: # mode == 'adjacency'
        return A[i,j]


def scaleProbabilities(probs):
    return list(np.array(probs)/float(np.sum(np.array(probs))))


def computeNegativeProbabilities(A):
    """
    Compute probabilities for 'simple' negative sampling
    """
    import networkx as nx
    
    G = nx.from_numpy_array(A)
    return scaleProbabilities([nx.degree(G,nbunch=i) for i in range(len(A))])


def getNodeSimilarity(load_path, flag, A, mode='adjacency', n_walks=10, walk_length=10, window_size=5, p=.25, q=4, n_negative=5, features=None):
    """
    Wrapper of function computing similarity between node pairs
    
    Inputs:
        A Adjacency matrix of a graph
        MODE Similarity metric between nodes
        NODES List of nodes to be used (if one graph is split in training and test sets)
        FEATURES Features of graph nodes (used for some similarity metrics)
        
    Outputs:
        NODE_PAIRS List of node pairs, for which similarity was computed
        SIMILARITY Measure of similarity between two nodes
    """
    if mode=='randomwalk': # node2vec implementation
        from RandomWalks import getRWSimilarity
        target, context, similarity, neg_samples = getRWSimilarity(load_path, flag, A, n_walks, walk_length, window_size, p, q, n_negative)
    else: # adjacency
        node_pairs = np.array(list(permutations(range(len(A)), 2)))
        np.random.shuffle(node_pairs)
        target = node_pairs[:,0]
        context = node_pairs[:,1]
#        if n_negative > 0:
#            neg_samples = np.zeros((len(node_pairs), n_negative), dtype=int)
#            neg_probs = computeNegativeProbabilities(A)
#        else:
        n_negative = 0
        neg_samples = None
    
        similarity = np.zeros((len(node_pairs),n_negative+1))
        similarity[:,0] = [NodeSimilarity(A,i,j,mode) for i,j in node_pairs]
#        if n_negative > 0:
#            for pair in range(len(node_pairs)):
#                neg_samples[pair,:] = np.random.choice(range(node_pairs[pair,1])+range(node_pairs[pair,1]+1,len(A)), size=n_negative, replace=True, p=scaleProbabilities(neg_probs[:node_pairs[pair,1]]+neg_probs[node_pairs[pair,1]+1:]))
#                similarity[pair,1:] = np.array([NodeSimilarity(A,node_pairs[pair,1],ns,mode) for ns in neg_samples[pair,:]]).reshape((n_negative,))
        
    return target, context, similarity, neg_samples

