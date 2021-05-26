
'''from tensorflow.keras import backend as K
import random

from transform import *
'''
from tensorflow.keras import backend as K
import random
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

import os
from image_loading import *
import argparse
from transform import *


def data_generator(train_classes, X_train, batch_size):
    classes_list = list(set(train_classes))
    #print('ciao', X_train)
    #print('lista',list(X_train))

    while True:
        a = []
        p = []
        n = []
        for _ in range(batch_size):
            pos_neg= random.sample(classes_list,2)
            positive_samples= random.sample(list(X_train[train_classes == pos_neg[0]]),2)
            #print('type:',type(positive_samples))
            #print('positive_sample', positive_samples)
            negative_samples = random.choice(list(X_train[train_classes == pos_neg[1]]))
            a.append(positive_samples[0])
            p.append(positive_samples[1])
            n.append(negative_samples)

        yield ([np.array(a), np.array(p), np.array(n)], np.zeros((batch_size,1 )).astype("float32"))

def triplet_loss(y_true, y_pred):
    print('len',y_pred.shape) #(none,300)
    anchor_out = y_pred[:, 0: 100]
    positive_out = y_pred[:, 100: 200]
    negative_out = y_pred[:, 200: 300]
    pos_dist = K.sum(K.abs(anchor_out - positive_out), axis=1)
    neg_dist = K.sum(K.abs(anchor_out - negative_out), axis=1)
    probs = K.softmax([pos_dist, neg_dist], axis=1)
    return K.mean(K.abs(probs[0]) + K.abs(1.0 - probs[1]))












