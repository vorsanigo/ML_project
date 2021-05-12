import os
import numpy as np
from sklearn.neighbors import NearestNeighbors

from image_loading import read_imgs_dir
from autoencoder import AutoEncoder
from transform import apply_transformer, modifyImage
from visualization import *


training = True

# Make paths
TrainDir = os.path.join(os.getcwd(), "data", "train")
TestDir = os.path.join(os.getcwd(), "data", "test")
OutputDir = os.path.join(os.getcwd(), "output", "convAE")
if not os.path.exists(OutputDir):
    os.makedirs(OutputDir)

# Augment the datasets
print("\nAugmentig dataset")
modifyImage(TrainDir)

# Read images
extensions = [".jpg", ".jpeg", ".png"]
print("\nLoading training images")
imgs_train = read_imgs_dir(TrainDir, extensions)
print("Loading test images")
imgs_test = read_imgs_dir(TestDir, extensions)
shape_img = imgs_train[0].shape
print("\nImage shape = " + str(shape_img))

# Build models
autoencoderFile = os.path.join(OutputDir, "ConvAE_autoecoder.h5")
encoderFile = os.path.join(OutputDir, "ConvAE_encoder.h5")

model = AutoEncoder(shape_img, autoencoderFile, encoderFile)
model.set_arch()

input_shape_model = tuple([int(x) for x in model.encoder.input.shape[1:]])
output_shape_model = tuple([int(x) for x in model.encoder.output.shape[1:]])
n_epochs = 500

# Normalize all images
print("\nNormalizing training images")
imgs_train = apply_transformer(imgs_train)
print("Normalizing test images")
imgs_test = apply_transformer(imgs_test)

# Convert images to numpy array of right dimensions
print("\nConverting to numpy array of right dimensions")
X_train = np.array(imgs_train).reshape((-1,) + input_shape_model)
X_test = np.array(imgs_test).reshape((-1,) + input_shape_model)
print(">>> X_train.shape = " + str(X_train.shape))
print(">>> X_test.shape = " + str(X_test.shape))

# Train (if necessary)
if training:
    print("\nStart training...")
    model.compile(loss="mse", optimizer="adam")
    model.fit2(X_train, n_epochs=n_epochs, batch_size=256)
    model.save_models()
    print("Done training")
else:
    print("\nLoading model...")
    model.load_models(loss="mse", optimizer="adam")


# Create embeddings using model
print("\nCreating embeddings")
E_train = model.predict(X_train)
E_train_flatten = E_train.reshape((-1, np.prod(output_shape_model)))
E_test = model.predict(X_test)
E_test_flatten = E_test.reshape((-1, np.prod(output_shape_model)))
print(">>> E_train.shape = " + str(E_train.shape))
print(">>> E_test.shape = " + str(E_test.shape))
print(">>> E_train_flatten.shape = " + str(E_train_flatten.shape))
print(">>> E_test_flatten.shape = " + str(E_test_flatten.shape))


# Fit kNN model on training images
print("\nFitting KNN model on training data...")
knn = NearestNeighbors(n_neighbors=5, metric="cosine")
knn.fit(E_train_flatten)
print("Done fitting")

# Querying on test images
print("\nQuerying...")

for i, emb_flatten in enumerate(E_test_flatten):
    distances, indx = knn.kneighbors([emb_flatten]) # find k nearest train neighbours
    print("\nFor query image_" + str(i))
    print(">> Indices:" + str(indx))
    print(">> Distances:" + str(distances))
    img_query = imgs_test[i]  # query image
    imgs_retrieval = [imgs_train[idx] for idx in indx.flatten()]  # retrieval images
    outFile = os.path.join(OutputDir, "ConvAE_retrieval_" + str(i) + ".png")
    plot_query_retrieval(img_query, imgs_retrieval, None)
