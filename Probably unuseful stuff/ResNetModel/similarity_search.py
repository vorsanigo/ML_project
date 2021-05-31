from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
#matplotlib inline       # Show the plots as a cell within the Jupyter Notebooks

#filenames = pickle.load(open('filenames-caltech101.pickle', 'rb'))
#feature_list = pickle.load(open('features-caltech101-resnet.pickle', 'rb'))
feature_list = pickle.load(open('list_tot_features_training.pickle', 'rb'))
print(feature_list)
images_paths_training = pickle.load(open('images_paths_training.pickle', 'rb'))
print(images_paths_training)
images_paths_gallery = pickle.load(open('images_paths_gallery.pickle', 'rb'))
print(images_paths_gallery)


'''for i in filenames:
    print(i)'''


neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute',metric='euclidean').fit(feature_list)
distances, indices = neighbors.kneighbors([feature_list[0]])
print(indices)
plt.imshow(mpimg.imread(images_paths[0]))
plt.show()

for i in range(5):
    print(distances[0][i])

plt.imshow(mpimg.imread(images_paths[indices[0]]))
plt.show()

'''for i in range(6):
    random_image_index = random.randint(0,num_images)
    distances, indices = neighbors.kneighbors([featureList[random_image_index]])
    # don't take the first closest image as it will be the same image
    similar_image_paths = [filenames[random_image_index]] +
                          [filenames[indices[0][i]] for i in range(1,4)]
    plot_images(similar_image_paths, distances[0])'''