import os
import numpy as np
from sklearn.neighbors import NearestNeighbors

from image_loading import Loader
from autoencoder import AutoEncoder
from transform import normalize_img, data_augmentation
from visualization import *
from scipy import spatial
import argparse



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
                    default=100,
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



'''wandb.login()

# trigger or untrigger WandB
if args.wandb == 'False' or args.mode == 'deploy':
    os.environ['WANDB_MODE'] = 'dryrun'
'''
# 1. Start a W&B run
'''wandb.init(project='aml-challenge', 
           )
wandb.config.epochs = args.e
wandb.config.batch_size = args.bs'''

'''wandb.init(project='aml-challenge',
           entity='innominati',
           group=args.mode,
           name=args.n,
           config={  # and include hyperparameters and metadata
               #"learning_rate": args.lr,
               "epochs": args.e,
               "batch_size": args.bs,
         })
config = wandb.config
'''


#training = True

# Make paths
TrainDir = os.path.join(os.getcwd(), args.data_path, "training")
QueryDir = os.path.join(os.getcwd(), args.data_path, "validation", "query")
GalleryDir = os.path.join(os.getcwd(), args.data_path, "validation", "gallery")
OutputDir = os.path.join(os.getcwd(), "output", "convAE")
if not os.path.exists(OutputDir):
    os.makedirs(OutputDir)
# TODO MODIFIED USING PARSER
'''TrainDir = os.path.join(os.getcwd(), "dataset2", "training")
QueryDir = os.path.join(os.getcwd(), "dataset2", "validation", "query")
GalleryDir = os.path.join(os.getcwd(), "dataset2", "validation", "gallery")
OutputDir = os.path.join(os.getcwd(), "output", "convAE")
if not os.path.exists(OutputDir):
    os.makedirs(OutputDir)'''

# Augment the datasets
#print("\nAugmentig dataset")
#modifyImage(TrainDir)



# Read images
loader = Loader(args.img_size, args.img_size, args.channels)
# TODO MODIFIED USING PARSER
#loader = Loader(100, 100, 3)
train_map = loader.get_files(TrainDir)
train_paths, imgs_train, train_classes = loader.get_data_paths(train_map)
query_map = loader.get_files(QueryDir)
query_paths, imgs_query, query_classes = loader.get_data_paths(query_map)
gallery_map = loader.get_files(GalleryDir)
gallery_paths, imgs_gallery, gallery_classes = loader.get_data_paths(gallery_map)

shape_img = imgs_train[0].shape  # bc we need it as argument for the Autoencoder()
print(shape_img)


# Build models
autoencoderFile = os.path.join(OutputDir, "ConvAE_autoecoder.h5")
encoderFile = os.path.join(OutputDir, "ConvAE_encoder.h5")

model = AutoEncoder(shape_img, autoencoderFile, encoderFile)
model.set_arch()

input_shape_model = tuple([int(x) for x in model.encoder.input.shape[1:]])
output_shape_model = tuple([int(x) for x in model.encoder.output.shape[1:]])
#n_epochs = args.e # non serve perché lo passiamo direttamente nel modello
# TODO MODIFIED USING PARSER
#n_epochs = 500

# Normalize all images
print("\nNormalizing training images")
imgs_train = normalize_img(imgs_train)
print("Normalizing query images")
imgs_query = normalize_img(imgs_query)
print("Normalizing gallery images")
imgs_gallery = normalize_img(imgs_gallery)


# Convert images to numpy array of right dimensions
print("\nConverting to numpy array of right dimensions")
X_train = np.array(imgs_train).reshape((-1,) + input_shape_model)
X_query = np.array(imgs_query).reshape((-1,) + input_shape_model)
X_gallery = np.array(imgs_gallery).reshape((-1,) + input_shape_model)
print(">>> X_train.shape = " + str(X_train.shape))
print(">>> X_query.shape = " + str(X_query.shape))
print(">>> X_gallery.shape = " + str(X_gallery.shape))

# Creare object for train augmentation
trainGen = data_augmentation(X_train, args.bs)
#trainGen = X_train

# Train (if necessary)
if args.mode == "training model":
    print("\nStart training...")
    model.compile(loss=args.loss, optimizer="adam")
    model.fit2(trainGen, n_epochs=args.e, batch_size=args.bs)
    model.save_models()
    print("Done training")
else:
    print("\nLoading model...")
    model.load_models(loss=args.loss, optimizer="adam")
# TODO MODIFIED WITH PARSER AND DATA AUGMENTATION
'''if training:
    print("\nStart training...")
    model.compile(loss="mse", optimizer="adam")
    model.fit2(X_train, n_epochs=n_epochs, batch_size=256)
    model.save_models()
    print("Done training")
else:
    print("\nLoading model...")
    model.load_models(loss="mse", optimizer="adam")'''


# Create embeddings using model
print("\nCreating embeddings")
E_train = model.predict(X_train)
E_train_flatten = E_train.reshape((-1, np.prod(output_shape_model)))
E_query = model.predict(X_query)
E_query_flatten = E_query.reshape((-1, np.prod(output_shape_model)))
E_gallery = model.predict(X_gallery)
E_gallery_flatten = E_gallery.reshape((-1, np.prod(output_shape_model)))
print(">>> E_train.shape = " + str(E_train.shape))
print(">>> E_query.shape = " + str(E_query.shape))
print(">>> E_gallery.shape = " + str(E_gallery.shape))
print(">>> E_train_flatten.shape = " + str(E_train_flatten.shape))
print(">>> E_query_flatten.shape = " + str(E_query_flatten.shape))
print(">>> E_gallery_flatten.shape = " + str(E_gallery_flatten.shape))

# define the distance between query - gallery features vectors
pairwise_dist = spatial.distance.cdist(E_query_flatten, E_gallery_flatten, args.metric, p=2.)
# TODO MODIFIED WITH PARSER
#pairwise_dist = spatial.distance.cdist(E_query_flatten, E_gallery_flatten, 'minkowski', p=2.)
# rows -> queries | columns -> gallery --> cell = distance between query-gallery image
print('\nComputed distances and got c-dist {}'.format(pairwise_dist.shape))

print("\nCalculating indices...")
indices = np.argsort(pairwise_dist, axis=-1)
print("Indices: {}".format(indices))

'''
Distances:
[1.06049268 0.98174144 0.84297278 1.18097723 1.33711798 0.7725198
  1.21793345 0.6474991  1.55152428 1.477141   1.25295738 1.28248735
  1.7081946  1.93887704 1.20129754 1.51035105 1.62115751 1.45156932
  1.29350864 2.17163186 2.34592395 2.0875376  1.50529649 1.98142459
  2.05400083 2.32826204 1.72161598 1.62639113]

Indices:
[ 7  5  2  1  0  3 14  6 10 11 18  4 17  9 22 15  8 16 27 12 26 13 23 24
  21 19 25 20]
  
Interpretation:
"From "Indices" --> the element in position 7 in the "Distances" array is the smallest"
bc 7 is in first position.
So indices are sorted so that the relative distances are sorted from smallest (index 7) to largest (index 20)
'''

print("\nGallery classes", gallery_classes)
gallery_matches = gallery_classes[indices]
print("\nMatches")
print(gallery_matches)


def topk_accuracy(gt_label, matched_label, k=1):
    matched_label = matched_label[:, :k]
    total = matched_label.shape[0]
    correct = 0
    for q_idx, q_lbl in enumerate(gt_label):
        correct+= np.any(q_lbl == matched_label[q_idx, :]).item()
    acc_tmp = correct/total

    return acc_tmp


print('\n########## RESULTS ##########')

for k in [1, 3, 10]:
    topk_acc = topk_accuracy(query_classes, gallery_matches, k)
    print('>>> Top-{:d} Accuracy: {:.3f}'.format(k, topk_acc))




# Fit kNN model on training images
print("\nFitting KNN model on training data...")
knn = NearestNeighbors(n_neighbors=5, metric="cosine")
knn.fit(E_gallery_flatten)
print("Done fitting")

# Querying on test images
print("\nQuerying...")

for i, emb_flatten in enumerate(E_query_flatten):
    distances, indx = knn.kneighbors([emb_flatten])  # find k nearest gallery neighbours
    print("\nFor query image_" + str(i))
    print(">> Indices:" + str(indx))
    print(">> Distances:" + str(distances))
    img_query = imgs_query[i]  # query image
    imgs_retrieval = [imgs_gallery[idx] for idx in indx.flatten()]  # retrieval images
    outFile = os.path.join(OutputDir, "ConvAE_retrieval_" + str(i) + ".png")
    #plot_query_retrieval(img_query, imgs_retrieval, None)
