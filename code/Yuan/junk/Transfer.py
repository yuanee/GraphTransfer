#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:18:13 2019

@author: yuanneu
"""

def keras_sg_second(trainlist,weight1,weight2):
    N,d=weight1.shape
    negative_num=trainlist[0][2].shape[1]
    shared_layer1 = Embedding(input_dim=N, output_dim=d, weights=[weight1])
    #shared_layer1 is the output layer
    shared_layer2 = Embedding(input_dim=N, output_dim=d, weights=[weight2])
    #shared_layer2 is the hidden layer
    input_target = Input(shape=(1,), dtype='int32', name='input_1')
    input_source = Input(shape=(1,), dtype='int32', name='input_2')
    input_negative = Input(shape=(negative_num,),dtype='int32',name='input_3')
    target= shared_layer1(input_target)
    source= shared_layer2(input_source)
    negative= shared_layer1(input_negative)
    positive_dot = dot([source, target], axes=(2), normalize=False)
    negative_dot = dot([source, negative], axes=(2), normalize=False)
    all_dot = concatenate([positive_dot, negative_dot],axis=2)
    sigmoid_sample = Activation('sigmoid')(all_dot)
    
    model = Model(inputs=[input_target,input_source,input_negative], outputs=[sigmoid_sample])
    sgd2 = optimizers.SGD(lr=0.025, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd2)
    for [a1,a2,a4,y1] in trainlist:
        loss = model.train_on_batch([a1, a2, a4], y1)
    embed_output=shared_layer1.get_weights()[0]
    embed_hidden=shared_layer2.get_weights()[0]
    return embed_output,embed_hidden