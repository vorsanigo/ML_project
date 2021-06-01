import argparse
import os
from scipy import spatial
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from triplets import triplet_loss
from autoencoder import AutoEncoder, TripletsEncoder
from image_loading import read_imgs_no_subfolders
from transform import normalize_img, data_augmentation
from visualization import plot_query_retrieval
from final_display import *
import numpy as np
import time


parser = argparse.ArgumentParser(description='Description challenge test')
parser.add_argument('--data_path',
                    '-d',
                    type=str,
                    default='dataset_test',
                    help='Dataset path')
parser.add_argument('-img_size',
                    type=int,
                    default=324,
                    help='image size for the model')
parser.add_argument('-channels',
                    type=int,
                    default=3,
                    help='number of channels')
parser.add_argument('-model',
                    type=str,
                    default='convAE',
                    help='Default model = convAE, other options: pretrained, triplets_loss')
parser.add_argument('-plot',
                    type=str,
                    default='False',
                    help='Helper to visualize the results (default = False)')
parser.add_argument('-metric',
                    type=str,
                    default='minkowski',
                    help='metric to compute distance query-gallery')

args = parser.parse_args()

shape_img = (args.img_size, args.img_size, args.channels)

QueryDir = os.path.join(os.getcwd(), args.data_path, "query")
GalleryDir = os.path.join(os.getcwd(), args.data_path, "gallery")
OutputDir = os.path.join(os.getcwd(), "output", args.model)
if not os.path.exists(OutputDir):
    os.makedirs(OutputDir)

QueryImgs, QueryName = read_imgs_no_subfolders(QueryDir, args.img_size)
GalleryImgs, GalleryName = read_imgs_no_subfolders(GalleryDir, args.img_size)
QueryName = [os.path.split(img_path)[1] for img_path in QueryName]
GalleryName = [os.path.split(img_path)[1] for img_path in GalleryName]

# Normalize all images
print("Normalizing query images")
QueryImgs = normalize_img(QueryImgs)
print("Normalizing gallery images")
GalleryImgs = normalize_img(GalleryImgs)

if args.model == 'convAE':

    # Build models
    autoencoderFile = os.path.join(OutputDir, "ConvAE_autoecoder.h5")
    print("autoencoder file", autoencoderFile)
    encoderFile = os.path.join(OutputDir, "ConvAE_encoder.h5")
    model = AutoEncoder(shape_img, autoencoderFile, encoderFile)
    model.set_arch()

    input_shape_model = tuple([int(x) for x in model.encoder.input.shape[1:]])
    output_shape_model = tuple([int(x) for x in model.encoder.output.shape[1:]])

    # Loading model
    model.load_models(loss='mse', optimizer="adam")
    
    # Convert images to numpy array of right dimensions
    print("\nConverting to numpy array of right dimensions")
    X_query = np.array(QueryImgs).reshape((-1,) + input_shape_model)
    X_gallery = np.array(GalleryImgs).reshape((-1,) + input_shape_model)
    print(">>> X_query.shape = " + str(X_query.shape))
    print(">>> X_gallery.shape = " + str(X_gallery.shape))
    
    # Create embeddings using model
    print("\nCreating embeddings")
    E_query = model.predict(X_query)
    E_query_flatten = E_query.reshape((-1, np.prod(output_shape_model)))
    E_gallery = model.predict(X_gallery)
    E_gallery_flatten = E_gallery.reshape((-1, np.prod(output_shape_model)))



    print("\nComputing pairwise distance between query and gallery images")

    # Define the distance between query - gallery features vectors
    pairwise_dist = spatial.distance.cdist(E_query_flatten, E_gallery_flatten, args.metric, p=2.)
    print('\nComputed distances and got c-dist {}'.format(pairwise_dist.shape))

    print("\nCalculating indices and gallery matches...")
    indices = np.argsort(pairwise_dist, axis=-1)

    final_res_pairwise = dict()
    for i, emb_flatten in enumerate(indices):
        img_query = QueryImgs[i]
        query_name = QueryName[i]
        imgs_retrieval = [GalleryImgs[indx] for indx in indices[i][:10]]
        names_retrieval = [GalleryName[indx] for indx in indices[i][:10]]

        if args.plot == 'True':
            outFile = os.path.join(OutputDir, "ConvAE_retrieval_pairwise_" + str(i) + ".png")
            plot_query_retrieval(img_query, imgs_retrieval, None)

        create_results_dict(final_res_pairwise, query_name, names_retrieval)


    print("\nComputing knn for distance between query and gallery images")

    # Fit kNN model on gallery images
    print("\nFitting KNN model on training data...")
    k = 10
    knn = NearestNeighbors(n_neighbors=k, metric="cosine")
    knn.fit(E_gallery_flatten)
    print("Done fitting")

    # Querying on test images
    final_res_knn = dict()
    print("\nQuerying...")
    for i, emb_flatten in enumerate(E_query_flatten):
        distances, indx = knn.kneighbors([emb_flatten])
        img_query = QueryImgs[i]  
        query_name = QueryName[i]
        imgs_retrieval = [GalleryImgs[idx] for idx in indx.flatten()]
        names_retrieval = [GalleryName[idx] for idx in indx.flatten()]

        if args.plot == 'True':
            outFile = os.path.join(OutputDir, "ConvAE_retrieval_knn_" + str(i) + ".png")
            plot_query_retrieval(img_query, imgs_retrieval, outFile)

        create_results_dict(final_res_knn, query_name, names_retrieval)

    print('Saving results...')
    final_results_pairwise = create_final_dict(final_res_pairwise)
    final_results_knn = create_final_dict(final_res_knn)
    url = "http://ec2-18-191-24-254.us-east-2.compute.amazonaws.com/results/"
    #submit(final_results_pairwise, url)
    submit(final_results_knn, url)
    print("Done saving")


elif args.model == 'pretrained':

    # Load pre-trained VGG19 model + higher level layers
    print("\nLoading model...")
    model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=shape_img)
    model.summary()

    shape_img_resize = tuple([int(x) for x in model.input.shape[1:]])
    input_shape_model = tuple([int(x) for x in model.input.shape[1:]])
    output_shape_model = tuple([int(x) for x in model.output.shape[1:]])

    # Convert images to numpy array of right dimensions
    print("\nConverting to numpy array of right dimensions")
    X_query = np.array(QueryImgs).reshape((-1,) + input_shape_model)
    X_gallery = np.array(GalleryImgs).reshape((-1,) + input_shape_model)
    print(">>> X_query.shape = " + str(X_query.shape))
    print(">>> X_gallery.shape = " + str(X_gallery.shape))

    # Create embeddings using model
    print("\nCreating embeddings")
    E_query = model.predict(X_query)
    E_query_flatten = E_query.reshape((-1, np.prod(output_shape_model)))
    E_gallery = model.predict(X_gallery)
    E_gallery_flatten = E_gallery.reshape((-1, np.prod(output_shape_model)))



    print("\nComputing pairwise distance between query and gallery images")

    # Define the distance between query - gallery features vectors
    pairwise_dist = spatial.distance.cdist(E_query_flatten, E_gallery_flatten, args.metric, p=2.)
    print('\nComputed distances and got c-dist {}'.format(pairwise_dist.shape))

    print("\nCalculating indices and gallery matches...")
    indices = np.argsort(pairwise_dist, axis=-1)

    final_res_pairwise = dict()
    for i, emb_flatten in enumerate(indices):
        img_query = QueryImgs[i]
        query_name = QueryName[i]
        imgs_retrieval = [GalleryImgs[indx] for indx in indices[i][:10]]
        names_retrieval = [GalleryName[indx] for indx in indices[i][:10]]

        if args.plot == 'True':
            outFile = os.path.join(OutputDir, "ConvAE_retrieval_pairwise_" + str(i) + ".png")
            plot_query_retrieval(img_query, imgs_retrieval, None)

        create_results_dict(final_res_pairwise, query_name, names_retrieval)


    print("\nComputing knn for distance between query and gallery images")

    # Fit kNN model on gallery images
    print("\nFitting KNN model on training data...")
    k = 10
    knn = NearestNeighbors(n_neighbors=k, metric="cosine")
    knn.fit(E_gallery_flatten)
    print("Done fitting")

    # Querying on test images
    final_res_knn = dict()
    print("\nQuerying...")
    for i, emb_flatten in enumerate(E_query_flatten):
        distances, indx = knn.kneighbors([emb_flatten])
        img_query = QueryImgs[i]  
        query_name = QueryName[i]
        imgs_retrieval = [GalleryImgs[idx] for idx in indx.flatten()]
        names_retrieval = [GalleryName[idx] for idx in indx.flatten()]
        
        if args.plot == 'True':
            outFile = os.path.join(OutputDir, "Pretr_retrieval_knn_" + str(i) + ".png")
            plot_query_retrieval(img_query, imgs_retrieval, outFile)

        create_results_dict(final_res_knn, query_name, names_retrieval)

    print('Saving results...')
    final_results_pairwise = create_final_dict(final_res_pairwise)
    final_results_knn = create_final_dict(final_res_knn)
    url = "http://ec2-18-191-24-254.us-east-2.compute.amazonaws.com/results/"
    #submit(final_results_pairwise, url)
    submit(final_results_knn, url)
    print("Done saving")

else:
    # Build model
    tripletsFile = os.path.join(OutputDir, "triplets_encoder.h5")
    triplet_model = TripletsEncoder(shape_img, tripletsFile)
    triplet_model.set_arch()

    # Convert images to numpy array of right dimensions
    print("\nConverting to numpy array of right dimensions")
    X_query = np.array(QueryImgs).reshape((-1,) + shape_img)
    X_gallery = np.array(GalleryImgs).reshape((-1,) + shape_img)

    # Loading model
    triplet_model.load_triplets(triplet_loss, optimizer="adam")

    # Create embeddings using model
    print("\nCreating embeddings")
    E_query = triplet_model.predict_triplets(X_query)
    E_gallery = triplet_model.predict_triplets(X_gallery)


    print("\nComputing pairwise distance between query and gallery images")

    # Define the distance between query - gallery features vectors
    pairwise_dist = spatial.distance.cdist(E_query, E_gallery, args.metric, p=2.)
    print('\nComputed distances and got c-dist {}'.format(pairwise_dist.shape))

    print("\nCalculating indices and gallery matches...")
    indices = np.argsort(pairwise_dist, axis=-1)

    final_res_pairwise = dict()
    for i, emb_flatten in enumerate(indices):
        img_query = QueryImgs[i]
        query_name = QueryName[i]
        imgs_retrieval = [GalleryImgs[indx] for indx in indices[i][:10]]
        names_retrieval = [GalleryName[indx] for indx in indices[i][:10]]

        if args.plot == 'True':
            outFile = os.path.join(OutputDir, "ConvAE_retrieval_pairwise_" + str(i) + ".png")
            plot_query_retrieval(img_query, imgs_retrieval, None)

        create_results_dict(final_res_pairwise, query_name, names_retrieval)


    print("\nComputing knn for distance between query and gallery images")

    # Fit kNN model on gallery images
    print("\nFitting KNN model on training data...")
    k = 10
    knn = NearestNeighbors(n_neighbors=k, metric="cosine")
    knn.fit(E_gallery)
    print("Done fitting")

    # Querying on test images
    final_res_knn = dict()
    print("\nQuerying...")
    for i, emb_flatten in enumerate(E_query):
        distances, indx = knn.kneighbors([emb_flatten])
        img_query = QueryImgs[i]
        query_name = QueryName[i]
        imgs_retrieval = [GalleryImgs[idx] for idx in indx.flatten()]
        names_retrieval = [GalleryName[idx] for idx in indx.flatten()]
        if args.plot == 'True':
            outFile = os.path.join(OutputDir, "Triplets_retrieval_knn_" + str(i) + ".png")
            plot_query_retrieval(img_query, imgs_retrieval, None)

        create_results_dict(final_res_knn, query_name, names_retrieval)

    print('Saving results...')
    final_results_pairwise = create_final_dict(final_res_pairwise)
    final_results_knn = create_final_dict(final_res_knn)
    url = "http://ec2-18-191-24-254.us-east-2.compute.amazonaws.com/results/"
    #submit(final_results_pairwise, url)
    submit(final_results_knn, url)
    print("Done saving")
