#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse
import os
from glob import glob
import pickle


# https://stackoverflow.com/questions/1038160/data-structure-for-maintaining-tabular-data-in-memory
def query_lod(lod, filter=None, sort_keys=None):
    if filter is not None:
        lod = (r for r in lod if filter(r))
    if sort_keys is not None:
        lod = sorted(lod, key=lambda r:[r[k] for k in sort_keys])
    else:
        lod = list(lod)
    return lod


def lookup_lod(lod, **kw):
    res = []
    idx = []
    for ind in range(len(lod)):
        row = lod[ind]
        for k,v in kw.iteritems():
            if row[k] != str(v): break
        else:
            res.append(row)
            idx.append(ind)
    return res, idx


def csv_import(results, filename):
    import csv
    csv_columns = ['Dataset', 'Cliq', 'Labels', 'Features', 'Embedding', 'TopSim', 'EmbSim', 'SimLoss', 'TransMode', 'GraphDist', 'Epochs', 'Convergence', 'TrainAcc', 'TestAccA', 'RsquaredA', 'TestAccB', 'RsquaredB']
    
    try:
        with open(filename+'.csv', 'w') as handler:
            writer = csv.DictWriter(handler, fieldnames=csv_columns, delimiter='\t')
            writer.writeheader()
            for result in results:
                writer.writerow(result)
#        print("CSV file with global results has been saved. Path: " + filename + ".csv") 
    except IOError:
        print("I/O error when saving global results to CSV file") 


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Process results of Graph Transfer Learning Neural Network", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--process', type=str, choices=['parse','query'], help='Parse results (create PICKLE, TXT and CSV files), otherwise outputs results from PICKLE file for a given query')
    
    parser.add_argument('-p', '--filepath', type=str, default='/home/$USER/GraphTransferLearning/results/', help='Full path to file with results')
    parser.add_argument('-n', '--filename', type=str, default='GlobalResults', help='Name of files to load/create')
    
    parser.add_argument('-d', '--dataset', type=str, default='synthetic', help='Specify dataset to be used')
    parser.add_argument('--cliq', type=int, default=4, help='Specify number of node clusters for synthetic dataset')
    parser.add_argument('--labels', type=str, default='cluster', help='Node labels of the synthetic  data')
    parser.add_argument('--features', type=str, default='index', help='Node features of the synthetic data')
    parser.add_argument('-et', '--embedding', type=str, default='word2vec', help='Type of embedding function: word2vec, 1nn')
    parser.add_argument('-sg', '--topology_similarity', type=str, default='adjacency', help='Similarity measure between nodes of the same graph in graph topological space')
    
    parser.add_argument('-se', '--embedding_similarity', type=str, default='innerprod', help='Similarity measures between nodes of the same graph in embedding space')
    parser.add_argument('-sl', '--similarity_loss', type=str, default='l2', help='Loss function between similarity in topological space and similarity in embedding space for nodes of the same graph')
    
    parser.add_argument('--transmode', type=str, default='1graph', help='Algorithm computing next value of P matrix')
    parser.add_argument('-gd', '--graph_distance', type=str, default='l2', help='Distance measure used to compute all pairwise distances between nodes of two graphs in the common embedding space (matrix D)')
    
    args = parser.parse_args()
#    args.filepath='/Users/agritsenko/Dropbox/Research/NEU-SPIRAL/GraphTransferLearning/results/'
    fullpath = os.path.join(args.filepath, args.filename)
    
    if args.process == 'parse':
        # Parse results and create TXT, PICKLE and CSV files        
        results = []
        Dataset = 'synthetic'
        with open(fullpath + '.txt', 'w') as file:
            
            labels = sorted(glob(args.filepath + '*/'))
            labels = labels[::-1]
            for label in labels[::-1]:
                Label = label[label.rfind('/',0,-1)+1:-1]
                file.write("\n{:*^245s}".format(' '+Label.upper()+' '))
                Cliq = Label[Label.find('-')+1:]
                Label = Label[:Label.find('-')]
                if Label.lower() == 'cluster':
                    file.write('\n' + ' '*15 + '\t::\tConv\tTrain\tTestA\t:\tConv\tTrain\tTestA\tTestB\t:\tConv\tTrain\tTestA\tTestB\t:\tConv\tTrain\tTestA\tTestB\t:\tConv\tTrain\tTestA\tTestB\t:\tConv\tTrain\tTestA\tTestB')
                else:
                    file.write('\n' + ' '*15 + '\t::\tConv\tTrain\tTestA\tRsqrdA\t:\tConv\tTrain\tTestA\tRsqrdA\tTestB\tRsqrdB\t:\tConv\tTrain\tTestA\tRsqrdA\tTestB\tRsqrdB\t:\tConv\tTrain\tTestA\tRsqrdA\tTestB\tRsqrdB\t:\tConv\tTrain\tTestA\tRsqrdA\tTestB\tRsqrdB\t:\tConv\tTrain\tTestA\tRsqrdA\tTestB\tRsqrdB')
                
                methods = sorted(glob(label + '*/'))
                for method in methods:
                    Method = method[method.rfind('/',0,-1)+1:-1]
                    output = "\n{:^15s}\t:".format(Method)
                    
                    if Method == 'node2vec':
                        Embedding = 'skipgram'
                        TopSim = 'randomwalk'
                        EmbSim = 'softmax'
                        SimLoss = 'crossentropy'
                    elif Method == 'factorization':
                        Embedding = 'unified'
                        TopSim = 'adjacency'
                        EmbSim = 'l2'
                        SimLoss = 'innerprod'
                    elif Method == 'eigenmaps':
                        Embedding = 'unified'
                        TopSim = 'adjacency'
                        EmbSim = 'innerprod'
                        SimLoss = 'l2'
                    
                    modes = sorted(glob(method + '*/'))
                    modes = [modes[i] for i in [0,2,4,5,1,3]]
                    for mode in modes:
                        filename = sorted(glob(mode+'Accuracy*.txt'), key=os.path.getmtime)[0]
                        with open(filename, 'r') as handle:
                            accuracy = handle.readline()
                        accuracy = accuracy.split()
                        
                        Mode = mode[mode.rfind('/',0,-1)+1:-1]
                        Epochs = int(filename[filename.rfind('_',0,-1)+1:filename.rfind('.',0,-1)])
                        Convergence = float(accuracy[0])
                        Train = float(accuracy[1])
                        TestA = float(accuracy[2])
                        outputline = "{0:>.1f}\t{1:.2f}\t{2:.2f}".format(Convergence, Train, TestA)
                        R2A, TestB, R2B = 'N/A', 'N/A', 'N/A'
                        if Label != 'cluster':
                            R2A = float(accuracy[3])
                            outputline += "\t{:>4.2f}".format(R2A)
                        if Mode != '1graph':
                            if Label != 'cluster':
                                TestB = float(accuracy[4])
                                R2B = float(accuracy[5])
                                outputline += "\t{0:.2f}\t{1:>4.2f}".format(TestB, R2B)
                            else:
                                TestB = float(accuracy[3])
                                outputline += "\t{:.2f}".format(TestB)
                        new_res = {'Dataset':Dataset, 'Cliq':Cliq, 'Labels':Label, 'Features':'index', 'Embedding':Embedding, 'TopSim':TopSim, 'EmbSim':EmbSim, 'SimLoss':SimLoss, 'TransMode':Mode, 'GraphDist':'l2', 'Epochs':Epochs, 'Convergence':Convergence, 'TrainAcc':Train, 'TestAccA':TestA, 'RsquaredA':R2A, 'TestAccB':TestB, 'RsquaredB':R2B}
                        results.append(new_res)
                        
                        output += ":\t" + outputline + "\t"
                    file.write(output)
                file.write("\n")
        
        with open(fullpath + '.pkl', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
#            print('Save results to csv file')
        results = query_lod(results, sort_keys=('Dataset', 'Cliq', 'Labels', 'Features', 'Embedding', 'TopSim', 'EmbSim', 'SimLoss', 'TransMode', 'GraphDist'))
        csv_import(results, fullpath)
    
    elif args.process == 'query':
        # Process query and output results from PICKLE file
        try:
            with open(fullpath + '.pkl', 'rb') as handle:
                results = pickle.load(handle)
#            print('Print results with respect to query')
            lod, _ = lookup_lod(results, Dataset=args.dataset, Labels=args.labels, Features=args.features, Embedding=args.embedding, TopSim=args.topology_similarity, EmbSim=args.embedding_similarity, SimLoss=args.similarity_loss, P=args.Popt, GraphDist=args.graph_distance)
            if len(lod)<1:
                print("No records have been found for a given experiment")
            elif len(lod)==1:
                print("After %5d epoch(s):\tTrain accuracy = %.4f\tTest accuracy = %.4f" % (lod[0]['Epochs'], lod[0]['TrainAcc'], lod[0]['TestAcc']))
            else:
                print(lod)
        except IOError:
            print("I/O error when reading file '" + fullpath + ".pkl'")
        except EOFError:
            print("File '" + fullpath + ".pkl' is empty")
    
    
    
    
    
    
    
    
    