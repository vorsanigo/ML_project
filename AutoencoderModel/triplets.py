from tensorflow.keras import backend as K
import random
from transform import *
from tensorflow.keras import backend as K
import random
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import os
from image_loading import *
import argparse
from transform import *


def data_generator_1(train_classes, x_train, batch_size):

    """ This function generates triplets to use for the triplets loss """

    classes_list = list(set(train_classes))

    while True:
        a = []
        p = []
        n = []
        for _ in range(batch_size):
            pos_neg = random.sample(classes_list, 2)
            positive_samples = random.sample(list(x_train[train_classes == pos_neg[0]]), 2)
            # print('type:',type(positive_samples))
            # print('positive_sample', positive_samples)
            negative_samples = random.choice(list(x_train[train_classes == pos_neg[1]]))
            a.append(positive_samples[0])
            p.append(positive_samples[1])
            n.append(negative_samples)

        yield [np.array(a), np.array(p), np.array(n)], np.zeros((batch_size, 1)).astype("float32")


        
def data_generator(train_classes, X_train, batch_size):
    # train classes retrieved by loader
    # defined in main at row 104
    classes_list = list(set(train_classes))
    # X_train = list(X_train)
    # while True:
    for _ in range(batch_size):
        a = []
        p = []
        n = []
        pos_neg = random.sample(classes_list, 2)
        positive_samples = random.sample(list(X_train[train_classes == pos_neg[0]]), 2)
        negative_samples = random.choice(list(X_train[train_classes == pos_neg[1]]))
        a.append(positive_samples[0])
        p.append(positive_samples[1])
        n.append(negative_samples)
        print("A", a)
        trainGenAnchor, trainGenPositive, trainGenNegative = data_augmentation_triplet(np.array(a), np.array(p),
                                                                                       np.array(n), batch_size)
        print("DONEEEEEEEEEEE")
        # yield ([np.array(a), np.array(p), np.array(n)], np.zeros((batch_size, 1)).astype("float32"))
        while True:
            Xa = trainGenAnchor.next()
            Xp = trainGenPositive.next()
            Xn = trainGenNegative.next()
            yield [Xa[0], Xp[0], Xn[0]], Xa[1]


            
def triplet_loss(y_true, y_pred):

    """ This function returns the triplets loss """

    # print('len', y_pred.shape)  # (none,300)
    anchor_out = y_pred[:, 0: 100]
    positive_out = y_pred[:, 100: 200]
    negative_out = y_pred[:, 200: 300]
    pos_dist = K.sum(K.abs(anchor_out - positive_out), axis=1)
    neg_dist = K.sum(K.abs(anchor_out - negative_out), axis=1)
    probs = K.softmax([pos_dist, neg_dist], axis=1)
    return K.mean(K.abs(probs[0]) + K.abs(1.0 - probs[1]))












