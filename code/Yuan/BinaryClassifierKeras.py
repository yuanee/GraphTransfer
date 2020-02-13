


import numpy as np
import networkx as nx
import tensorflow as tf
import math
import random
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from keras.utils import np_utils
import sys
import pickle
import argparse
from keras.layers import Input, Embedding, Dense, Lambda, Reshape, Activation
from keras.models import Model
from keras import optimizers
from keras.layers.merge import concatenate, dot
from keras import backend as K
from keras.models import Sequential
import keras



def BinaryKeras(Num,dim):
    model = Sequential()
    model.add(Dense(dim, input_dim=Num, kernel_initializer='normal',use_bias=False, name='embedding',activation=None))
    model.add(Dense(1,activation='sigmoid',name='weight', kernel_initializer='zeros',  bias_initializer='zeros'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def modify_weight(model,weight):   
    model.get_layer("embedding").set_weights([weight])
    return model

