import pickle
import os
import glob
#import cv2
import numpy as np
import random
import tensorflow as tf
from PIL import Image

def load_stl(data_path = '/home/veror/Desktop/Uni/Applied Machine Learning/prova resnet/stl10', norm=True, gray=False):
    def normalize(data):
        data = (data / 255) - 0.5
        return data

    def load_data(mypath, gray):
        if gray:
            data = np.zeros((0, 96, 96, 1), dtype='uint8')
        else:
            data = np.zeros((0, 96, 96, 3), dtype='uint8')
        labels = np.zeros((0,))
        for i, cla in enumerate(mypath):
            filelist = glob.glob(os.path.join(cla, '*.png'))
            if gray:
                tmp_data = np.empty((len(filelist), 96, 96, 1), dtype='uint8')
            else:
                tmp_data = np.empty((len(filelist), 96, 96, 3), dtype='uint8')
            tmp_labels = np.ones((len(filelist),)) * i
            for j, path in enumerate(filelist):
                if gray:
                    image = Image.open(path).convert('L')
                    image = np.asarray(image)
                    image = np.expand_dims(image, axis=-1)
                else:
                    image = Image.open(path)
                    image = np.asarray(image)
                tmp_data[j, :] = image
                print(image.shape)
            data = np.concatenate((data, tmp_data))
            labels = np.concatenate((labels, tmp_labels))
        return data, labels

    train_path = glob.glob(os.path.join(data_path, 'train', '*'))
    train_path.sort()
    test_path = glob.glob(os.path.join(data_path, 'test', '*'))
    test_path.sort()
    training_data, training_labels = load_data(train_path, gray)
    test_data, test_labels = load_data(test_path, gray)
    perm = np.arange(test_data.shape[0])
    random.shuffle(perm)
    perm = perm[:1000]
    test_data = test_data[perm, :]
    test_labels = test_labels[perm]
    if norm:
        training_data = normalize(training_data)
        test_data = normalize(test_data)
    return training_data, training_labels, test_data, test_labels


def load_cifar10():
    def unpickle_cifar10(path):
        files = glob.glob(path)
        od = np.zeros((0, 32, 32, 3), dtype='uint8')
        ol = np.zeros((0,))
        for i in files:
            with open(i, 'rb') as fo:
                mydict = pickle.load(fo, encoding='bytes')
            data = mydict[b'data']
            data = data.reshape(data.shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
            labels = np.asarray(mydict[b'labels'])
            od = np.concatenate((od, data))
            ol = np.concatenate((ol, labels))
        return od, ol
    data_path = '/data/datasets/cifar10/cifar-10-batches-py'
    training_data, training_labels = unpickle_cifar10(os.path.join(data_path, 'data*'))
    test_data, test_labels = unpickle_cifar10(os.path.join(data_path, 'test*'))
    return training_data, training_labels, test_data, test_labels


def load_cifar100():
    def unpickle(file):
        with open(file, 'rb') as fo:
            mydict = pickle.load(fo, encoding='bytes')
        data = mydict[b'data']
        data = data.reshape(data.shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
        labels = np.asarray(mydict[b'coarse_labels'])
        return data, labels
    data_path = '/data/datasets/cifar-100-python'
    training_data, training_labels = unpickle(os.path.join(data_path, 'train'))
    test_data, test_labels = unpickle(os.path.join(data_path, 'test'))
    return training_data, training_labels, test_data, test_labels

load_stl()
'''def load_visual_small(data_path='/data/datasets/visual_small'):
    def load_data(mypath):
        data = np.zeros((0, 150, 150, 3), dtype='uint8')
        labels = np.zeros((0,))
        for i, cla in enumerate(mypath):
            filelist = glob.glob(os.path.join(cla, '*.jpg'))
            tmp_data = np.empty((len(filelist), 150, 150, 3), dtype='uint8')
            tmp_labels = np.ones((len(filelist),)) * i
            for j, path in enumerate(filelist):
                image = cv2.imread(path)
                image = cv2.resize(image, (150, 150))
                tmp_data[j, :] = image
            data = np.concatenate((data, tmp_data))
            labels = np.concatenate((labels, tmp_labels))
        return data, labels
    train_path = glob.glob(os.path.join(data_path, 'train', '*'))
    train_path.sort()
    test_path = glob.glob(os.path.join(data_path, 'test', '*'))
    test_path.sort()
    training_data, training_labels = load_data(train_path)
    test_data, test_labels = load_data(test_path)
    return training_data, training_labels, test_data, test_labels'''