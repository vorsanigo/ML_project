import os
import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from image_loading import Loader
from transform import normalize_img
from final_display import *
from scipy import spatial
import argparse
import wandb


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
                    default='False',
                    help='Log on WandB (default = False)')
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
parser.add_argument('-plot',
                    type=str,
                    default='True',
                    help='Helper to visualize the results (default = True)')
args = parser.parse_args()

wandb.login(key='f97918185ed02886a90fa4464e7469c13c017460')

# trigger or untrigger WandB
if args.wandb == 'False' or args.mode == 'deploy':
    os.environ['WANDB_MODE'] = 'dryrun'

# 1. Start a W&B run
wandb.init(project='aml-challenge',
           )
wandb.config.epochs = args.e
wandb.config.batch_size = args.bs

# Make paths
TrainDir = os.path.join(os.getcwd(), args.data_path, "training")
QueryDir = os.path.join(os.getcwd(), args.data_path, "validation", "query")
GalleryDir = os.path.join(os.getcwd(), args.data_path, "validation", "gallery")
OutputDir = os.path.join(os.getcwd(), "output", "convAE")
if not os.path.exists(OutputDir):
    os.makedirs(OutputDir)

# create loader
loader = Loader(args.img_size, args.img_size, args.channels)

# extract img final for the model
shape_img = (args.img_size, args.img_size, args.channels)  # bc we need it as argument for the Autoencoder()
print("Image size", shape_img)

# Read images
query_map = loader.get_files(QueryDir)
query_names, query_paths, imgs_query, query_classes = loader.get_data_paths(query_map)
gallery_map = loader.get_files(GalleryDir)
gallery_names, gallery_paths, imgs_gallery, gallery_classes = loader.get_data_paths(gallery_map)

# Load pre-trained VGG19 model + higher level layers
if args.mode != "training model":
    print("\nLoading model...")
    model = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=shape_img)
    model.summary()

shape_img_resize = tuple([int(x) for x in model.input.shape[1:]])
input_shape_model = tuple([int(x) for x in model.input.shape[1:]])
output_shape_model = tuple([int(x) for x in model.output.shape[1:]])

# Normalize all images
print("Normalizing query images")
imgs_query = normalize_img(imgs_query)
print("Normalizing gallery images")
imgs_gallery = normalize_img(imgs_gallery)

# Convert images to numpy array of right dimensions
print("\nConverting to numpy array of right dimensions")
X_query = np.array(imgs_query).reshape((-1,) + input_shape_model)
X_gallery = np.array(imgs_gallery).reshape((-1,) + input_shape_model)
print(">>> X_query.shape = " + str(X_query.shape))
print(">>> X_gallery.shape = " + str(X_gallery.shape))

# Create embeddings using model
print("\nCreating embeddings")
E_query = model.predict(X_query)
E_query_flatten = E_query.reshape((-1, np.prod(output_shape_model)))
E_gallery = model.predict(X_gallery)
E_gallery_flatten = E_gallery.reshape((-1, np.prod(output_shape_model)))

# define the distance between query - gallery features vectors
pairwise_dist = spatial.distance.cdist(E_query_flatten, E_gallery_flatten, args.metric, p=2.)
# rows -> queries | columns -> gallery --> cell = distance between query-gallery image
print('\nComputed distances and got c-dist {}'.format(pairwise_dist.shape))

print("\nCalculating indices...")
indices = np.argsort(pairwise_dist, axis=-1)

gallery_matches = gallery_classes[indices]

def topk_accuracy(gt_label, matched_label, k=1):
    matched_label = matched_label[:, :k]
    total = matched_label.shape[0]
    correct = 0
    for q_idx, q_lbl in enumerate(gt_label):
        correct += np.any(q_lbl == matched_label[q_idx, :]).item()
    acc_tmp = correct/total

    return acc_tmp


print('\n########## RESULTS ##########')

for k in [1, 3, 10]:
    topk_acc = topk_accuracy(query_classes, gallery_matches, k)
    print('>>> Top-{:d} Accuracy: {:.3f}'.format(k, topk_acc))


# Fit kNN model on training images
print("\nFitting KNN model on training data...")
k = 10
knn = NearestNeighbors(n_neighbors=k, metric="cosine")
knn.fit(E_gallery_flatten)
print("Done fitting")

# Querying on test images
final_res = dict()
print("\nQuerying...")
for i, emb_flatten in enumerate(E_query_flatten):
    distances, indx = knn.kneighbors([emb_flatten])
    img_query = imgs_query[i]  
    query_name = query_names[i]
    imgs_retrieval = [imgs_gallery[idx] for idx in indx.flatten()]
    names_retrieval = [gallery_names[idx] for idx in indx.flatten()]
    
    if args.plot == 'True':
        outFile = os.path.join(OutputDir, "Pretr_retrieval_" + str(i) + ".png")
        plot_query_retrieval(img_query, imgs_retrieval, outFile)

    create_results_dict(final_res, query_name, names_retrieval)

print('Saving results')
final_results = create_final_dict(final_res)
url = "http://kamino.disi.unitn.it:3001/results/"

