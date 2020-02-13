#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:22:31 2019

@author: Andrey Gritsenko
         SPIRAL Group
         Electrical & Computer Engineering
         Northeastern University
"""

"""
Main module to run GTL model. Depending on the input command line parameters, it loads certain data, creates specific GTL model, trains it on train data, tests on test data, and outputs results
"""

import argparse
import os
import numpy as np

from LoadData import loadData
from createGTLmodel import getGTLmodel
from trainGTLmodel import train
from SaveData import saveGlobalResults


def str2bool(v):
    if v.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif v.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

if __name__ == '__main__':
    
    debug = False 
    if not debug:
        parser = argparse.ArgumentParser(description="Train Graph Transfer Learning model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
        # dataset parameters
        parser.add_argument('-lp', '--load_path', type=str, default='', 
                            help='Full path to folder with datasets')
        parser.add_argument('-d', '--dataset', type=str, default='synthetic', 
                            choices=['synthetic', 'zachary', 'disease', 'email', 'facebook'],
                            help='Dataset to be used')
        
        # general graph parameters
        parser.add_argument('--features', type=str, default='index', 
                            choices=['index', 'onehot', 'neighbor', 'heat'],
                            help='Node features of the synthetic data')
        parser.add_argument('-nl', '--ncliq', type=int, default=4, choices=[4,6],
                            help='Number of clusters in synthetic data graphs')
        parser.add_argument('--labels', type=str, default='cluster', 
                            choices=['cluster', 'infection', 'pagerank'],
                            help='Node labels of the synthetic data')
        parser.add_argument('--depth', type=int, default=1,
                            help='Number of hidden layers in Prediction Branch')
        
        # graph embedding parameters
        parser.add_argument('--nembedding', type=int, default=5,
                            help='Size of the output embedding vector')
        parser.add_argument('-sg', '--topology_similarity', type=str, default='randomwalk', 
                            choices=['randomwalk', 'adjacency'],
                            help='Similarity measure between nodes of the same graph in graph topological space')
        parser.add_argument('-et', '--embedding_type', type=str, default='skipgram', 
                            choices=['unified', 'skipgram'],
                            help='Type of embedding function: skipgram, unified')
        parser.add_argument('-se', '--embedding_similarity', type=str, default='softmax', 
                            choices=['softmax', 'innerprod', 'cossim', 'l2'],
                            help='Similarity measures between nodes of the same graph in embedding space')
        parser.add_argument('-sl', '--similarity_loss', type=str, default='crossentropy', 
                            choices=['crossentropy', 'innerprod', 'l2'],
                            help='Loss function between similarity in topological space and similarity in embedding space for nodes of the same graph')
        parser.add_argument('-prl', '--prediction_loss', type=str, default='mean_squared_error',
                            help='Loss function for prediction branch')
        parser.add_argument('-af', '--activation_function', type=str, default='tanh',
                            choices=['tanh', 'sigmoid', 'relu'],
                            help='Activation function for prediction branch neurons')
        
        # randomwalk parameters
        parser.add_argument('--nwalks', type=int, default=20,
                            help='Number of node2vec random walks')
        parser.add_argument('--walk_length', type=int, default=10,
                            help='Length of random walk')
        parser.add_argument('--window_size', type=int, default=4,
                            help='Width of sliding window in random walks')
        parser.add_argument('--p', type=float, default=0.25, 
                            help='Parameter p for node2vec random walks')
        parser.add_argument('--q', type=float, default=4.0, 
                            help='Parameter q for node2vec random walks')
        parser.add_argument('--nnegative', type=int, default=5, 
                            help='Number of negative samples used in skip-gram')
        parser.add_argument('--scale_negative', type=str2bool, default=False,
                            help='Specifies whether to scale outputs for negative samples') # possible solution to batch size issue
        
        # second graph parameters
        parser.add_argument('--transfer_mode', type=str, default='1graph',
                            choices=['1graph', 'noP', 'iterP', 'optP', 'trueP', 'trueP_DS'],
                            help='Specifies transfer learning mode')
        parser.add_argument('--b_from_a', type=str, choices=['permute','modify','split'], default='permute', 
                            help='Specifies whether to permute, add/remove edges, or split first graph to generate 2nd graph')
        parser.add_argument('-sw', '--same_weights', type=str2bool, default=False,
                            help='Specifies whether A and B embedding branches will be initialized with the same weights')
        parser.add_argument('-gd', '--graph_distance', type=str, default='l2', 
                            choices=['l2', 'innerprod', 'cossim'],
                            help='Pairwise distance measure between nodes in the embedding space (matrix D)')
        
        # neural net train/test parameters
        parser.add_argument('--alpha', type=float, default=1.0,
                            help='Weight of graph matching loss')
        parser.add_argument('--beta', type=str2bool, default=False,
                            help='Specifies whether to scale parts of P-optimization loss')
        parser.add_argument('-lr', '--learning_rate', type=float, default=0.025,
                            help='Learning rate')
        parser.add_argument('--batch_size', type=int, default=2, 
                            help='Number of instances in each batch')
        parser.add_argument('--epochs', type=int, default=2, 
                            help='Number of epochs')
        parser.add_argument('--early_stopping', type=int, default=0, 
                            help='Number of epochs with no improvement after which training will be stopped. If <=0, no early stopping is used')
        parser.add_argument('--iterations', type=int, default=1,
                            help='Number of iterations for model to initialize and run. Ouput results are averaged across iterations')
        parser.add_argument('--id_gpu', default=-1, type=int, 
                            help='Specifies which gpu to use. If <0, model is run on cpu')
        
        # results parameters
        parser.add_argument('-sp', '--save_path', type=str, default='', 
                            help='Full path to folder where results are saved')
        parser.add_argument('-v', '--visualize', action='store_true', 
                            help='Specifies whether to visualize model architecture or not')
        
        args = parser.parse_args()
    
        if args.id_gpu >= 0:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            # The GPU id to use
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)
    else:
        args = argparse.Namespace()
        args.load_path = '/Users/agritsenko/Dropbox/Research/NEU-SPIRAL/GraphTransferLearning/datasets/'
        args.dataset = 'synthetic'
        args.features = 'index'
        args.ncliq = 4
        args.labels = 'infection'
        args.depth = 1
        args.topology_similarity = 'adjacency' # 'randomwalk'
        args.nwalks = 5
        args.walk_length = 10
        args.window_size = 5
        args.p = 0.25
        args.q = 4.0
        args.embedding_type = 'unified' # 'skipgram'
        args.nembedding = 5
        args.nnegative = 5
        args.scale_negative = False
        args.embedding_similarity = 'innerprod' # 'l2' # 'softmax' # 
        args.similarity_loss = 'l2' # 'innerprod' # 'crossentropy' # 
        args.prediction_loss = 'mean_squared_error'
        args.activation_function = 'tanh'
        args.transfer_mode = 'trueP'
        args.same_weights = False
        args.b_from_a = 'permute'
        args.graph_distance = 'l2'
        args.alpha = 1.0
        args.beta = False
        args.learning_rate = 0.025
        args.batch_size = 1
        args.epochs = 2
        args.early_stopping = 2
        args.iterations = 2
        args.save_path = '/Users/agritsenko/Dropbox/Research/NEU-SPIRAL/GraphTransferLearning/results/' + args.labels + '-' + str(args.ncliq) + '/'
        args.visualize = False
    
    # save configuration file
    n_iter = args.iterations
    save_path = args.save_path
    output = open(save_path + 'output.log', 'w')
    print("*************** Configuration ***************", file=output)
    with open(save_path + 'gtl.config', 'w') as handle:
        args_dic = vars(args)
        for arg, value in args_dic.items():
            line = arg + ' : ' + str(value)
            print(line, file=output)
            handle.write(line+'\n')
    print("*********************************************\n", file=output)
    
    # load data
    dataset = args.dataset.lower()
    if dataset == 'synthetic':
        load_path = args.load_path + dataset + '/' + str(args.ncliq) + '/'
    else:
        load_path = args.load_path + dataset + '/'
    features = args.features.lower()
    labels = args.labels.lower()
    transmode = args.transfer_mode
    B_from_A = args.b_from_a
    visualize = args.visualize
    n_nodes, n_features, n_labels, A, Afeatures, Alabels, Atrain, Atest, B, Bfeatures, Blabels = loadData(load_path, features, labels, transmode, B_from_A, visualize)
    
    if labels == 'pagerank':
        n_layers = max(10, args.depth)
    else:
        n_layers = max(1, args.depth)
    
    n_embedding = args.nembedding
    topology_similarity = args.topology_similarity
    embedding_type = args.embedding_type
    embedding_similarity = args.embedding_similarity
    if embedding_type == 'skipgram':
        embedding_similarity = 'softmax'
    if embedding_similarity == 'softmax':
        n_negative = args.nnegative
        scale_negative = args.scale_negative
    else:
        n_negative = 0
        scale_negative = False
    similarity_loss = args.similarity_loss
    prediction_loss = args.prediction_loss
    activation_function = args.activation_function
    
    same_weights = args.same_weights
    graph_distance = args.graph_distance
    n_walks = args.nwalks
    walk_length = args.walk_length
    window_size = args.window_size
    p = args.p
    q = args.q
    
    alpha = args.alpha
    beta = args.beta
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    early_stopping = args.early_stopping
    
    n_epochs = args.epochs
    Aembedding, Alabels_output, Alabels_predicted, Bembedding, Blabels_output, Blabels_predicted, P = {}, {}, {}, {}, {}, {}, {}
    results = {'epochs':{}, 'train':{}, 'testA':{}}
    if labels != 'cluster':
        results['rsquaredA'] = {}
    if transmode != '1graph':
        results['testB'] = {}
        if labels != 'cluster':
            results['rsquaredB'] = {}
    for iter in range(n_iter):
        save_path_iter = save_path + str(iter+1) + '/'
        if not os.path.exists(save_path_iter):
            os.makedirs(save_path_iter)
        # create model
        models = getGTLmodel(n_nodes, n_features, n_embedding, labels, n_labels, n_layers, n_negative, scale_negative, embedding_type, embedding_similarity, similarity_loss, prediction_loss, activation_function, transmode, same_weights, graph_distance, learning_rate, alpha, save_path_iter, output, visualize)
        
        if iter==0:
            print_fn = lambda x: output.write(x + '\n')
            print("\nEmbedding model summary:".upper(), file=output)
            models['EmbeddingModel'].summary(print_fn=print_fn)
            print("\nEmbedding similarity branch summary:".upper(), file=output)
            models['EmbeddingModel'].get_layer('Branch_SimilarityA').summary(print_fn=print_fn)
            print("\nPrediction model summary:".upper(), file=output)
            models['PredictionModel'].summary(print_fn=print_fn)
            print("\nPrediction branch summary:".upper(), file=output)
            models['PredictionModel'].get_layer('Branch_Prediction').summary(print_fn=print_fn)
        
        print("\n ============================================== ", file=output)
        print("|*************** ITERATION #{0:3d} ***************|".format(iter), file=output)
        print(" ============================================== ", file=output)
        # train/test model
        suffix = '_' + str(n_epochs) + '_'
        iter_results = train(load_path, models, A, Afeatures, Alabels, Atrain, Atest, B, Bfeatures, Blabels, transmode, topology_similarity, n_walks, walk_length, window_size, p, q, n_negative, learning_rate, beta, n_epochs, early_stopping, batch_size, save_path_iter, suffix, output)
        results['epochs'][iter] = iter_results['epochs']
        results['train'][iter] = iter_results['acc_train']
        results['testA'][iter] = iter_results['acc_testA']
        if labels != 'cluster':
            results['rsquaredA'][iter] = iter_results['acc_rsquaredA']
        if transmode != '1graph':
            results['testB'][iter] = iter_results['acc_testB']
            if labels != 'cluster':
                results['rsquaredB'][iter] = iter_results['acc_rsquaredB']
    
        # save global results
        picklename = "GlobalResults"
        saveGlobalResults(args, results, picklename)
    
    print("\n ============================================== ", file=output)
    print("|*************** FINAL  RESULTS ***************|", file=output)
    print(" ============================================== \n", file=output)
    epochs_mean = np.mean(list(results['epochs'].values()))
    epochs_std = np.std(list(results['epochs'].values()))
    train_mean = np.mean(list(results['train'].values()))
    train_std = np.std(list(results['train'].values()))
    testA_mean = np.mean(list(results['testA'].values()))
    testA_std = np.std(list(results['testA'].values()))
    print("After {0:2d} iterations, the average\n\tConvergence rate = {1:.4f} (\u00B1{2:.4f})\n\tTrain accuracy (graph A) = {3:.4f} (\u00B1{4:.4f})\n\tTest accuracy (graph A) = {5:.4f} (\u00B1{6:.4f})".format(args.iterations, epochs_mean, epochs_std, train_mean, train_std, testA_mean, testA_std), file=output)
    if labels != 'cluster':
        rsquaredA_mean = np.mean(list(results['rsquaredA'].values()))
        rsquaredA_std = np.std(list(results['rsquaredA'].values()))
        print("\tR-squared (graph A) = {0:.4f} (\u00B1{1:.4f})".format(rsquaredA_mean, rsquaredA_std), file=output)
    if transmode != '1graph':
        testB_mean = np.mean(list(results['testB'].values()))
        testB_std = np.std(list(results['testB'].values()))
        print("\tTest accuracy (graph B) = {0:.4f} (\u00B1{1:.4f})".format(testB_mean, testB_std), file=output)
        if labels != 'cluster':
            rsquaredB_mean = np.mean(list(results['rsquaredB'].values()))
            rsquaredB_std = np.std(list(results['rsquaredB'].values()))
            print("\tR-squared (graph B) = {0:.4f} (\u00B1{1:.4f})".format(rsquaredB_mean, rsquaredB_std), file=output)
    output.close()
		
    
    
    
    
    
    
    
    
    
