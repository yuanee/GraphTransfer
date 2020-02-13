#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 22:00:01 2019

@author: yuanneu
"""



import numpy as np
import random
from keras.layers import Input, Embedding, Dense, Lambda, Reshape
from keras.models import Model
from keras import optimizers
from keras.layers.merge import concatenate, dot
from keras import backend as K



"""
main_input = Input(shape=(1,), dtype='int32', name='input_1')
main_input2 = Input(shape=(1,), dtype='int32', name='input_2')
main_input3 = Input(shape=(G.negative,), dtype='int32', name='input_3')
"""
# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
"""
embedding = np.random.uniform(-1.0/2.0/G.embedding_dimension, 1.0/2.0/G.embedding_dimension, (G.vocab_size, G.embedding_dimension))
shared_embedding_layer = Embedding(input_dim=G.vocab_size+, output_dim=G.embedding_dimension, weights=[embedding])
"""

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
            context_input=np.array([[context]])
            negative_samples=get_negative_sample(context)
            output_label=np.concatenate((np.array([[1.0]]), np.zeros((1,G.negative))),1)           
            yield ({'input_1': source_input, 'input_2': context_input, 'input_3': negative_samples}, {'output': output_label})
        

G=graph_G(10, 100, list(range(100)),8)

walks_sentences=[[1,2,3,4,5,6],[4,3,2,5,6,4]]

skip_input=skip_gram_generator(walks_sentences)
        
embedding=np.zeros((G.vocabulary,G.embedding_size))

shared_layer = Embedding(input_dim=G.vocabulary, output_dim=G.embedding_size, weights=[embedding])

main_input = Input(shape=(1,), dtype='int32', name='input_1')
main_input2 = Input(shape=(1,), dtype='int32', name='input_2')
main_input3 = Input(shape=(G.negative,),dtype='int32',name='input_3')

x1= shared_layer(main_input)
x2= shared_layer(main_input2)
x3= shared_layer(main_input3)

x4 = dot([x1, x2], axes=2, normalize=False)
x5 = dot([x1, x3], axes=2, normalize=False)
x6 = concatenate([x4, x5],axis=2)

x7 = Lambda(lambda t: K.mean(t,axis=1))(x6)

model = Model(inputs=[main_input,main_input2,main_input3], outputs=[x7])
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy')
model.fit_generator(skip_input, samples_per_epoch=1, nb_epoch=1)



"""
a1=np.array([[2]])
a2=np.array([[3]])
a4=np.array(([[3,4,5,5,6,7]]))
a3=np.tile(a1,[4,1])
y1=np.array([[0,1,1,0,1,0,1]])
y3=np.tile(y1,[4,1])
model.fit([a1,a2,a4], y1, epochs=10, batch_size=1)

#a2=np.array([[2,5]])
a_out=model.predict([a1,a2,a4])
print (a_out.shape)

"""
