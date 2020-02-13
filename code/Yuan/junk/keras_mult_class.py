from keras.layers import Input, Embedding, Dense, Lambda, Reshape, Activation
from keras.models import Model
from keras import optimizers
from keras.layers.merge import concatenate, dot



def keras_multiclass(trainlist,weight1,weight2):
    N,d=weight1.shape
    Nc,d=weight2.shape
    shared_layer1 = Embedding(input_dim=N, output_dim=d, weights=[weight1])
    shared_layer2 = Embedding(input_dim=Nc, output_dim=d, weights=[weight2])
    input_target = Input(shape=(1,), dtype='int32', name='input_target')
    input_negative = Input(shape=(Nc,),dtype='int32',name='input_beta')
    target= shared_layer1(input_target)
    beta= shared_layer2(input_negative)
    score_dot = dot([target, beta], axes=(2), normalize=False)
    sigmoid_out = Activation('softmax')(score_dot)
    print ('zero')
    model = Model(inputs=[input_target,input_negative], outputs=[sigmoid_out])
    sgd = optimizers.SGD(lr=0.025, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    
    for [a1,a2,y1] in trainlist:
        print ('zzt')
        loss2= model.train_on_batch([a1,a2],y1)
    embed_emb=shared_layer1.get_weights()[0]
    embed_beta=shared_layer2.get_weights()[0]
    return embed_emb,embed_beta
