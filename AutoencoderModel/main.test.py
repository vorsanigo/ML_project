import argparse
import os 
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf 
from autoencoder import AutoEncoder
from image_loading import read_imgs_no_subfolders
from transform import normalize_img, data_augmentation
from visualization import plot_query_retrieval
from final_display import *
import numpy as np 


parser = argparse.ArgumentParser(description='Description challenge test')
parser.add_argument('--data_path',
                    '-d',
                    type=str,
                    default='dataset',
                    help='Dataset path')
parser.add_argument('-img_size',
                    type=int,
                    default=324,
                    help='image size for the model')
parser.add_argument('-model',
                    type=str,
                    default='Autoencoder',
                    help='Default model = Autoencoder')

args = parser.parse_args()


QueryDir = os.path.join(os.getcwd(), args.data_path, "validation", "query")
GalleryDir = os.path.join(os.getcwd(), args.data_path, "validation", "gallery")
OutputDir = os.path.join(os.getcwd(), "output_{}".format(args.model))
if not os.path.exists(OutputDir):
    os.makedirs(OutputDir)
QueryImgs, QueryName =  read_imgs_no_subfolders(QueryDir)
GalleryImgs, GalleryName = read_imgs_no_subfolders(GalleryDir)
QueryName = [os.path.split(img_path)[1] for img_path in QueryName]
GalleryName = [os.path.split(img_path)[1] for img_path in GalleryName]

# Normalize all images
print("Normalizing query images")
QueryImgs = normalize_img(QueryImgs)
print("Normalizing gallery images")
GalleryImgs = normalize_img(GalleryImgs)



if args.model == 'Autoencoder':
    # Build models
    autoencoderFile = os.path.join(OutputDir, "ConvAE_autoecoder.h5")
    encoderFile = os.path.join(OutputDir, "ConvAE_encoder.h5")
    model = AutoEncoder(args.img_size, autoencoderFile, encoderFile)
    model.set_arch()

    # Convert images to numpy array of right dimensions
    input_shape_model = tuple([int(x) for x in model.encoder.input.shape[1:]])
    output_shape_model = tuple([int(x) for x in model.encoder.output.shape[1:]])

    model.load_models(loss = 'mse', optimizer = "adam")
    
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
        img_query = QueryImgs[i]  
        query_name = QueryName[i]
        imgs_retrieval = [GalleryImgs[idx] for idx in indx.flatten()]
        names_retrieval = [GalleryName[idx] for idx in indx.flatten()]
        if args.plot == 'True':
            outFile = os.path.join(OutputDir, "ConvAE_retrieval_" + str(i) + ".png")
            plot_query_retrieval(img_query, imgs_retrieval, outFile)
        create_results_dict(final_res, query_name, names_retrieval)

    final_results = create_final_dict(final_res)
    url = "http://kamino.disi.unitn.it:3001/results/"
    #submit(final_results, url)



elif  args.model == 'Pretrained':

    # Load pre-trained VGG19 model + higher level layers
    print("\nLoading model...")
    model = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=shape_img)
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
        img_query = QueryImgs[i]  
        query_name = QueryName[i]
        imgs_retrieval = [GalleryImgs[idx] for idx in indx.flatten()]
        names_retrieval = [GalleryName[idx] for idx in indx.flatten()]
        
        if args.plot == 'True':
            outFile = os.path.join(OutputDir, "Pretr_retrieval_" + str(i) + ".png")
            plot_query_retrieval(img_query, imgs_retrieval, outFile)

        create_results_dict(final_res, query_name, names_retrieval)

    print('Saving results')
    final_results = create_final_dict(final_res)
    url = "http://kamino.disi.unitn.it:3001/results/"

else:


    # Convert images to numpy array of right dimensions
    print("\nConverting to numpy array of right dimensions")
    X_query = np.array(QueryImgs).reshape((-1,) + input_shape_model)
    X_gallery = np.array(GalleryImgs).reshape((-1,) + input_shape_model)
    print(">>> X_query.shape = " + str(X_query.shape))
    print(">>> X_gallery.shape = " + str(X_gallery.shape))