
from tensorflow.keras import backend as K
import random
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

import os
from image_loading import *
import argparse
from transform import *


parser = argparse.ArgumentParser(description='Challenge presentation example')
parser.add_argument('--data_path',
                    '-d',
                    type=str,
                    default='dataset',
                    help='Dataset path')
parser.add_argument('-mode',
                    type=str,
                    default='training model',
                    help='training or test')
parser.add_argument('-n',
                    type=str,
                    default='training')
parser.add_argument('-lr',
                    type=float,
                    default=1e-4,
                    help='learning rate')
parser.add_argument('-e',
                    type=int,
                    default=50,
                    help='number of epochs')
parser.add_argument('-bs',
                    type=int,
                    default=32,
                    help='batch size')
parser.add_argument('-loss',
                    type=str,
                    default="mse",
                    help='loss function')
parser.add_argument('-wandb',
                    type=str,
                    default='True',
                    help='Log on WandB (default = True)')
parser.add_argument('-img_size',
                    type=int,
                    default=324,
                    help='image size for the model')
parser.add_argument('-channels',
                    type=int,
                    default=3,
                    help='number of channels')
parser.add_argument('-metric',
                    type=str,
                    default='minkowski',
                    help='metric to compute distance query-gallery')
args = parser.parse_args()




def data_generator(batch_size=64):
    classes_list = list(set(train_classes))
    while True:
        a=[]
        p=[]
        n=[]
        for _ in range(batch_size):
            pos_neg= random.sample(classes_list,2)
            positive_samples= random.choice(list(X_train[train_classes == pos_neg[0]]),2)
            negative_samples = random.choice(list(X_train[train_classes == pos_neg[1]]))
            a.append(positive_samples[0])
            p.append(positive_samples[1])
            n.append(negative_samples)
        yield ([np.array(a), np.array(p), np.array(n)], np.zeros((batch_size,1 )).astype("float32"))

def triplet_loss(y_true, y_pred):
    #[None, 300]
    #[[None, 300]*3]
    anchor_out = y_pred[:, 3528:7056]
    positive_out = y_pred[:, 3528: 7056]
    negative_out = y_pred[:, 7056:10584]
    pos_dist = K.sum(K.abs(anchor_out - positive_out), axis=1)
    neg_dist = K.sum(K.abs(anchor_out - negative_out), axis=1)
    probs = K.softmax([pos_dist, neg_dist], axis=1)
    return K.mean(K.abs(probs[0]) + K.abs(1.0 - probs[1]))

TrainDir = os.path.join(os.getcwd(), args.data_path, "training")

print('loading_images')
# create loader
loader = Loader(args.img_size, args.img_size, args.channels)

# extract img final for the model
shape_img = (args.img_size, args.img_size, args.channels)  # bc we need it as argument for the Autoencoder()
print("Image size", shape_img)

if args.mode == "training model":
    print('start_training')
    # TRAIN
    # Read images
    train_map = loader.get_files(TrainDir)
    train_names, train_paths, imgs_train, train_classes = loader.get_data_paths(train_map)

    # Normalize all images
    print("\nNormalizing training images")
    imgs_train = normalize_img(imgs_train)

    # Convert images to numpy array of right dimensions
    print("\nConverting to numpy array of right dimensions")
    X_train = np.array(imgs_train).reshape((-1,) + (324, 324, 3))
    print(">>> X_train.shape = " + str(X_train.shape))


# Set encoder

aaa = Input(shape=(324, 324, 3))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(aaa)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = Flatten()(x)
x = Dense(100, activation = 'relu')(x)
#encoded = MaxPooling2D((2, 2), padding='same')(x)

#modellll
model = Model(aaa, x)
model.summary()


triplet_model_a = Input((324,324,3))
triplet_model_p = Input((324,324,3))
triplet_model_n = Input((324,324,3))
triplet_model_out = Concatenate()([model(triplet_model_a),model(triplet_model_p),model(triplet_model_n)])
print('triplet_model_out', triplet_model_out)
triplet_model = Model([model(triplet_model_a),model(triplet_model_p),model(triplet_model_n)], triplet_model_out)
triplet_model.summary()

print('compile')
triplet_model.compile(loss=triplet_loss, optimizer = 'adam')

print('fitting')
triplet_model.fit_generator(data_generator(), steps_per_epoch=1, epochs = 3)

print('saving')
triplet_model.save('triplet.h5')

#model_embeddings= triplet_model.layers[3].predict(X_test, verbose=1)


