
import numpy as np
import networkx as nx
#import tensorflow as tf
#import math
#import random
#import time
#import pickle
from keras.layers import Input, Embedding, Activation
from keras.models import Model
from keras import optimizers
from keras.layers.merge import concatenate, dot
#import import_ipynb
#import node2vec.ipynb
from node2vec import Graph
#from ClassDataGen import *
from gensim.models import Word2Vec


def read_graph(Amatrix):
	'''
	Reads the input network in networkx.
	'''
	G=nx.from_numpy_matrix(Amatrix)	
	G = G.to_undirected()
	return G


def learn_embedding(Aa,pvalue,qvalue,num_walk,walk_length):
    """For a adjacency matrix Aa, generate random walk list by a return probability pvalue
    return probability qvalue, for each node a random walk has a length walk_length, each node has a total number of 
    'num_walk' walk. 
    """
    nx_G=read_graph(Aa)
    G=Graph(nx_G,False, pvalue,qvalue)
    G.preprocess_transition_probs()
    walks=G.simulate_walks(num_walk,walk_length)
    return walks


def word2vec(walks,window_size, Gn, dim, iteration):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
    This is the gensim package which can not initialize the embedding
	'''
	walks = [list(map(str, walk)) for walk in walks]
	model = Word2Vec(walks, size=dim, window=window_size, min_count=0, sg=1, negative=Gn, iter=iteration)
	return model



def frequency(walks):
    """ for a random walk, calculate the 3/4 occurrence probability for 
    negative sampling"""
    P_m={}
    for walk in walks:
        for item in walk:
            try: 
                P_m[item]+=1
            except:
                P_m[item]=1
    for key, value in P_m.items():
        P_m[key]=value**0.75
    return P_m


def negative_frequency(P_m):
    """get the negative probability"""
    sample_num=[]
    sample_prob=[]
    for key, value in P_m.items(): 
        sample_num.append(key)
        sample_prob.append(value)
    return sample_num, np.array(sample_prob)/sum(sample_prob)


def get_negative_sample(context,num,prob,Gn):
    """sample negative nodes for each context node"""
    negative_list=[]    
    while len(negative_list)<Gn:
        negative_sample = np.random.choice(num, p= prob.ravel())
        if (negative_sample!=context):
            negative_list.append(negative_sample)
        else:
            pass
    return np.array([negative_list])  




def skip_train(walks, window_size, Gn, iteration):
    """
    use the wallks to generate negative samples for neural network
    generate train input under the skip-gram formula
    """
    P_m=frequency(walks)
    Num,Prob=negative_frequency(P_m)
    p_dict=frequency(walks)        
    num, prob=negative_frequency(p_dict)    
    Trainlist=[]
    for itr in range(iteration):
        train_list=[]    
        for walk in walks:        
            for pod, word in enumerate(walk):  
                reduced_window = np.random.randint(window_size)
                source_input=np.array([[word]])
                start = max(0, pod - window_size+reduced_window)
                for pod2 in range(start, pod+window_size+1-reduced_window):
                    #print (pod2,pod)
                    if pod2!=pod:
                        try: 
                            target_input=np.array([[walk[pod2]]])
                            negative_input=get_negative_sample(target_input,Num,Prob,Gn)                        
                            output_label=np.concatenate((np.array([[1.0]]), np.zeros((1,Gn))),1)                         
                            train_list.append([target_input, source_input, negative_input, np.array([output_label])])
                        except:
                            pass
                    else:
                        pass
        Trainlist.extend(train_list)
    return Trainlist

def keras_sg_embedding(trainlist,weight1,weight2):
    #weight1, weight2 are Nxd numpy matrix 
    """The initial weights are weight1(output weight), weight2(hidden weight)
    the train input will update the weights by gradient descent"""
    N,d=weight1.shape
    negative_num=trainlist[0][2].shape[1]
    emb_target = Embedding(input_dim=N, output_dim=d, name='emb_target', weights=[weight1])
    #shared_layer1 is the output layer
    emb_source = Embedding(input_dim=N, output_dim=d, name='emb_source', weights=[weight2])
    #shared_layer2 is the hidden layer
    input_target = Input(shape=(1,), dtype='int32', name='input_target')
    input_source = Input(shape=(1,), dtype='int32', name='input_source')
    input_negative = Input(shape=(negative_num,),dtype='int32',name='input_negative')
    target = emb_target(input_target)
    source = emb_source(input_source)
    negative = emb_target(input_negative)
    positive_dot = dot([source, target], axes=(2), normalize=False)
    negative_dot = dot([source, negative], axes=(2), normalize=False)
    all_dot = concatenate([positive_dot, negative_dot],axis=2)
    sigmoid_sample = Activation('softmax')(all_dot)
    
    model = Model(inputs=[input_target,input_source,input_negative], outputs=[sigmoid_sample])
    model.summary()
    sgd = optimizers.SGD(lr=0.025, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd)
    
    ind = 0
    batch_size = len(trainlist)/20
    for [a1,a2,a4,y1] in trainlist:
        loss = model.train_on_batch([a1, a2, a4], y1)
#        print("Epoch %2d batch %4d: loss = %.4f" % (ind/batch_size + 1, ind%batch_size + 1, loss))
        ind += 1
    emb_target1 = emb_target.get_weights()[0]
    emb_source1 = emb_source.get_weights()[0]
    
    return emb_target1, emb_source1


"""
Nsize=26
dim=4 

A,B,ya,yb= GenDirtyGraph(Nsize,dim)
trainlistA,YtestA=GenerateM(ya) 
trA,testyA= splitB(ya) 

G=read_graph(A)

returnP=0.25
outputP=4
num_walk=10
walk_length=20
window_size=5
negative_num=4
dim=4
iteration=10
walks=learn_embedding(A,returnP,outputP,num_walk,walk_length)
trainlist=skip_train(walks,window_size,negative_num)
weight1=np.zeros((3,3))
weight2=np.zeros((3,3))
#emb1,emb2=keras_sg_embedding(trainlist,weight1,weight2)

model,wa=word2vec(walks,window_size, negative_num, dim, iteration)
Xfeature=[]
for n in range(Nsize):
    Xfeature.append(model.wv[str(n)])
    for j in range(Nsize):
        print (n,j, model.wv.similarity(str(n),str(j)))
Xa=np.array(Xfeature)
"""

"""
Nsize=26
dim=4 

A,B,ya,yb= GenDirtyGraph(Nsize,dim)


G=read_graph(A)

returnP=0.25
outputP=4
num_walk=10
walk_length=40
window_size=5
negative_num=4
dim=4
iteration=10
walks=learn_embedding(A,returnP,outputP,num_walk,walk_length)
trainlist=skip_train(walks,window_size,negative_num,iteration)
weight1=np.zeros((3,3))
weight2=np.zeros((3,3))
#emb1,emb2=keras_sg_embedding(trainlist,weight1,weight2)

model,wa=word2vec(walks,window_size, negative_num, dim, iteration)
Xfeature=[]
for n in range(Nsize):
    Xfeature.append(model.wv[str(n)]/np.linalg.norm(model.wv[str(n)]))
    for j in range(Nsize):
        print (n,j, model.wv.similarity(str(n),str(j)))
Xa=np.array(Xfeature)

weight1=np.random.multivariate_normal(np.zeros(dim), 0.1*np.identity(dim), Nsize)
weight2=np.random.multivariate_normal(np.zeros(dim), 0.1*np.identity(dim), Nsize)

weight1,weight2=keras_sg_embedding(trainlist,weight1,weight2)

Xfeature2=[]
for n in range(Nsize):
    Xfeature2.append(weight2[n,:]/np.linalg.norm(weight2[n,:]))
Xa2=np.array(Xfeature2)

Sa1=np.zeros((Nsize,Nsize))
Sa2=np.zeros((Nsize,Nsize))
for i in range(Nsize):
    for j in range(Nsize):
        Sa1[i,j]=np.dot(Xa[i,:],Xa[j,:])
        Sa2[i,j]=np.dot(Xa2[i,:],Xa2[j,:])
"""



