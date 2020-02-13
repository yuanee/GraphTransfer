#!/usr/bin/env python2
# -*- coding: utf-8 -*-


"""
Module provides visualization tools
"""

import os
import numpy as np
import pickle

from keras.utils import plot_model
from keras.utils.vis_utils import  model_to_dot

from processResults import lookup_lod


def plotArchitecture(model, file_name=None, save_path=None):
    """
    Visualizes Keras model. Depending on the parameters, model is either plotted in the console, 
    or saved to PNG file. If saved to PNG file, the file is created in the '/Models' subdirectory of 
    parent's directory. If the directory doesn't exist, it is created.
    
    Inputs:
        MODEL Keras model object to be visualized
        FILE_NAME Name of file where the model is saved. If FILE_NAME is not provided or empty, 
            the model is plotted in the console
    """
    
    if save_path == "" or save_path == None:
        path = os.path.realpath('..') + '/'
    else:
        path = save_path
    
    if model is not None:
        if file_name is not None and len(file_name.replace(" ", ""))>0:
#            print "Path when saving model is: " + path
            dir_path = path + 'Plots/'
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
            file_path = dir_path + file_name + '.png'
            plot_model(model, to_file=file_path, show_shapes=True)
            print('Model is saved to a file "' + file_path + '"')
        else:
            from IPython.display import SVG, display_svg
            print('Model architecture:')
            display_svg(SVG(model_to_dot(model).create(prog='dot', format='svg')))
    else:
        print("No model is provided")
    
        
def plotGraph(AdjMat, Labels, file_name=None, format="png", dpi=300, quality=None, transparent=True, frameon=False, save_path=None):
    """
    Visualized labeled graph
    
    Inputs:
        ADJMAT Adjacenccy matrix of a graph
        LABELS Set of node labels in one-hot coding format
        FILE_NAME Name of the file
        FORMAT File format, e.g. png, jpg, jpeg, pdf, svg, eps, ps
        DPI Image resolution in dots per inch
        QUALITY The image quality, on a scale from 1 (worst) to 95 (best). Applicable only if format is jpg or jpeg, ignored otherwise
        TRANSPARENT Transparency of axes and figure patches
        FRAMEON If True, the figure patch will be colored, if False, the figure background will be transparent
    """
    import networkx as nx
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    
    G = nx.from_numpy_matrix(AdjMat)
    Labels, idx = np.unique(np.argmax(Labels, axis=1), return_inverse=True) # Convert LABELS from one-hot coding format
    node_color = 1.*Labels[idx]/Labels.size + .5/Labels.size
    
    # Turn interactive plotting off
    plt.ioff() 
    fig = plt.figure()
    nx.draw(G, with_labels=True, node_color=node_color, cmap=cm.get_cmap(name='rainbow'))
    
    if save_path == "" or save_path == None:
        path = os.path.realpath('..') + '/'
    else:
        path = save_path
    
    if file_name is not None and len(file_name.replace(" ", ""))>0:
#        print "Path when saving graphs is: " + path
        dir_path = path + 'Plots/'
        if not os.path.isdir(dir_path): os.makedirs(dir_path)
        file_namepath = dir_path + file_name + '.' + format
        plt.savefig(file_namepath, format=format, dpi=dpi, quality=quality, transparent=transparent, frameon=frameon)
        print('Graph visualization is saved to a file "' + file_namepath + '"')
        plt.close(fig)
    else:
        plt.show()
        pass


# save results for each iteration
def saveIterationResults(save_path, suffix, Aembedding, Alabels_output, Alabels_predicted, Bembedding, Blabels_output, Blabels_predicted, P, acc_results):    
    if Alabels_predicted is not None:
        output_dict = dict(zip(['GraphAEmbedding', 'GraphALabels_outputs', 'GraphALabels_predicted'], [Aembedding, Alabels_output, Alabels_predicted]))
        if Bembedding is not None:
            output_dict.update(dict(zip(['GraphBEmbedding', 'GraphBLabels_outputs', 'GraphBLabels_predicted'], [Bembedding, Blabels_output, Blabels_predicted])))
    else:
        output_dict = dict(zip(['GraphAEmbedding', 'GraphALabels_outputs'], [Aembedding, Alabels_output]))
        if Bembedding is not None:
            output_dict.update(dict(zip(['GraphBEmbedding', 'GraphBLabels_outputs'], [Bembedding, Blabels_output])))
    if P is not None:
        output_dict.update(dict(zip(['P_predicted'], [P])))
    output_dict.update(dict(zip(['Accuracy'], [acc_results])))
    with open(save_path + 'results' + suffix + '.pkl', 'wb') as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


# add results to global results file
def saveGlobalResults(args, best_results, picklename):
    save_path = args.save_path[0:args.save_path.find('results/')+8]
    
    labels = args.labels
    transmode = args.transfer_mode
    
    epochs_mean = np.mean(list(best_results['epochs'].values()))
    epochs_std = np.std(list(best_results['epochs'].values()))
    train_mean = np.mean(list(best_results['train'].values()))
    train_std = np.std(list(best_results['train'].values()))
    testA_mean = np.mean(list(best_results['testA'].values()))
    testA_std = np.std(list(best_results['testA'].values()))
    
    with open(args.save_path + 'Accuracy_(' + str(args.iterations) + ')_' + str(args.epochs) + '.txt', "w") as f:
        if transmode == '1graph':
            if labels == 'cluster':
                f.write("%.4f (%.4f)\t%.4f (%.4f)\t%.4f (%.4f)\n" % (epochs_mean, epochs_std, train_mean, train_std, testA_mean, testA_std))
            else:
                rsquaredA_mean = np.mean(list(best_results['rsquaredA'].values()))
                rsquaredA_std = np.std(list(best_results['rsquaredA'].values()))
                f.write("%.4f (%.4f)\t%.4f (%.4f)\t%.4f (%.4f)\t%.4f (%.4f)\n" % (epochs_mean, epochs_std, train_mean, train_std, testA_mean, testA_std, rsquaredA_mean, rsquaredA_std))
        else:
            testB_mean = np.mean(list(best_results['testB'].values()))
            testB_std = np.std(list(best_results['testB'].values()))
            if labels == 'cluster':
                f.write("%.4f (%.4f)\t%.4f (%.4f)\t%.4f (%.4f)\t%.4f (%.4f)\n" % (epochs_mean, epochs_std, train_mean, train_std, testA_mean, testA_std, testB_mean, testB_std))
            else:
                rsquaredA_mean = np.mean(list(best_results['rsquaredA'].values()))
                rsquaredA_std = np.std(list(best_results['rsquaredA'].values()))
                rsquaredB_mean = np.mean(list(best_results['rsquaredB'].values()))
                rsquaredB_std = np.std(list(best_results['rsquaredB'].values()))
                f.write("%.4f (%.4f)\t%.4f (%.4f)\t%.4f (%.4f)\t%.4f (%.4f)\t%.4f (%.4f)\t%.4f (%.4f)\n" % (epochs_mean, epochs_std, train_mean, train_std, testA_mean, testA_std, rsquaredA_mean, rsquaredA_std, testB_mean, testB_std, rsquaredB_mean, rsquaredB_std))
    
    new_res = {'Dataset':args.dataset, 'Cliq':args.ncliq if args.dataset=='synthetic' else 'N/A', 'Labels':args.labels, 'Features':args.features, 'Embedding':args.embedding_type, 'TopSim':args.topology_similarity, 'EmbSim':args.embedding_similarity, 'SimLoss':args.similarity_loss, 'TransMode':transmode, 'GraphDist':args.graph_distance, 'Epochs_mean':epochs_mean, 'Epochs_std':epochs_std, 'TrainAcc_mean':train_mean, 'TrainAcc_std':train_std, 'TestAccA_mean':testA_mean, 'TestAccA_std':testA_std, 'RsquaredA_mean':rsquaredA_mean if labels!='cluster' else 'N/A', 'RsquaredA_std':rsquaredA_std if labels!='cluster' else 'N/A', 'TestAccB_mean':testB_mean if transmode!='1graph' else 'N/A', 'TestAccB_std':testB_std if transmode!='1graph' else 'N/A', 'RsquaredB_mean':rsquaredB_mean if (labels!='cluster' and transmode!='1graph') else 'N/A', 'RsquaredB_std':rsquaredB_std if (labels!='cluster' and transmode!='1graph') else 'N/A'}
    try:
        with open(save_path + picklename + '.pkl', 'rb') as handle:
            results = pickle.load(handle)
        _, idx = lookup_lod(results, Dataset=args.dataset, Cliq=args.ncliq, Labels=args.labels, Features=args.features, Embedding=args.embedding_type, TopSim=args.topology_similarity, EmbSim=args.embedding_similarity, SimLoss=args.similarity_loss, TransMode=transmode, GraphDist=args.graph_distance)
        if len(idx)==0:
            results.append(new_res)
        else:
            results[idx[0]] = new_res
        print("SAVING: Update an existing record")
    except:
        results = [new_res]
        print("SAVING: Create a new record")
    with open(save_path + picklename + '.pkl', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)





