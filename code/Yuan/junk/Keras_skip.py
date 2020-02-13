#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 22:00:01 2019

@author: yuanneu
"""



import numpy as np
import random
from keras.layers import Input, Embedding, Dense, Lambda, Reshape, Activation
from keras.models import Model
from keras import optimizers
from keras.layers.merge import concatenate, dot
from keras import backend as K


class graph_G():
    def __init__(self, realpart, vocabulary, vocabulary_list,embedding_size):
        self.negative = realpart
        self.vocabulary=vocabulary
        self.vocabulary_list = vocabulary_list
        self.embedding_size=embedding_size

       
def get_negative_sample(context):
	negative_samples = random.sample(G.vocabulary_list, G.negative)
	while context in negative_samples:
		negative_samples =random.sample(G.vocabulary_list, G.negative)
	return np.array([negative_samples])   

def skip_gram_generator(walks_sentences):
    for walk in walks_sentences:
        init_point=walk[0]
        source_input=np.array([[init_point]])
        for context in walk[1:]:
            target_input=np.array([[context]])
            negative_samples=get_negative_sample(context)
            output_label=np.concatenate((np.array([[1.0]]), np.zeros((1,G.negative))),1)           
            yield source_input, target_input, negative_samples, output_label
        

G=graph_G(10, 100, list(range(100)),8)

walks_sentences=[[1,2,3,4,5,6],[4,3,2,5,6,4]]
        
embedding=np.random.uniform(-1,1,(G.vocabulary,G.embedding_size))

shared_layer = Embedding(input_dim=G.vocabulary, output_dim=G.embedding_size, weights=[embedding])


input_target = Input(shape=(1,), dtype='int32', name='input_1')
input_source = Input(shape=(1,), dtype='int32', name='input_2')
input_negative = Input(shape=(G.negative,),dtype='int32',name='input_3')


target= shared_layer(input_target)
source= shared_layer(input_source)
negative= shared_layer(input_negative)


positive_dot = dot([target, source], axes=2, normalize=False)
negative_dot = dot([source, negative], axes=2, normalize=False)
all_dot = concatenate([positive_dot, negative_dot],axis=2)

all_dot = Lambda(lambda t: K.mean(t,axis=1))(all_dot)

sigmoid_sample = Activation('sigmoid')(all_dot)


model = Model(inputs=[input_target,input_source,input_negative], outputs=[sigmoid_sample])
model.compile(loss='binary_crossentropy', optimizer='rmsprop')


class_skip=skip_gram_generator(walks_sentences)
for i in range(8):
    (a1,a2,a4,y3)=next(class_skip)       
    loss = model.train_on_batch([a1, a2, a4], y3)
    #a_out=model.train_on_batch()
    print (loss)


