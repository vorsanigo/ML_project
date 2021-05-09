import numpy as np
from numpy.linalg import norm
import tensorflow.keras
from keras.preprocessing import image

import pickle
from tqdm import tqdm, tqdm_notebook
import os
import time

#reffered link: https://www.oreilly.com/library/view/practical-deep-learning/9781492034858/ch04.html

#from tensorflow.lite import image
#from tensorflow import keras
#from tensorflow import keras
#from tensorflow.keras.preprocessing.image import image
#from tf.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tqdm import notebook

'''Load the ResNet-50 model without the top classification layers, so we get only the bottleneck features. 
Then define a function that takes an image path, loads the image, resizes it to proper dimensions supported by ResNet-50, 
extracts the features, and then normalizes them:'''
import model

class FeatureExtractor:



    def extract_features(self, list_images, model): # 224, 3
        ''' def __init__(self, model):
                #self.feature_extractor = feature_extractor
                self.model = model'''

        '''input_shape = (img_size, img_size, num_channels)
        img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)'''
        #print(type(list_images[0]))
        print(list_images[0])
        print(list_images[0].shape)
        print(list_images[0].shape)
        tot_features = []
        for img in list_images:
            features = model.predict(img)
            flattened_features = features.flatten()
            normalized_features = flattened_features / norm(flattened_features)
            tot_features.append(normalized_features)
        return tot_features

#mod = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
'''mod = model.ResNet()
feat_ex = FeatureExtractor()
list_images = pickle.load(open('list_images_provae.pickle', 'rb'))
print(list_images[0].shape)
x = feat_ex.extract_features(list_images, mod)'''



#model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#img_path = '/home/veror/Desktop/Uni/Applied Machine Learning/prova resnet/Dataset/training/1/ec50k_00010001.jpg'



'''The function defined in the previous example is the key function 
that we use for almost every feature extraction need in Keras.'''

'''features = extract_features(img_path, model)
print(len(features)) #100352'''

'''The ResNet-50 model generated 2,048 features from the provided image. Each feature is a floating-point value between 0 and 1.'''
'''If your model is trained or fine tuned on a dataset that is not similar to ImageNet, 
redefine the “preprocess_input(img)” step accordingly. The mean values used in the function 
are particular to the ImageNet dataset. Each model in Keras has its own preprocessing function so 
make sure you are using the right one.'''


'''Now it’s time to extract features for the entire dataset. First, we get all the filenames with this handy function, 
which recursively looks for all the image files (defined by their extensions) under a directory:'''


extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
def get_file_list(root_dir):
    file_list = []
    counter = 1
    for root, directories, filenames in os.walk(root_dir):
        print('ciao')
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                file_list.append(os.path.join(root, filename))
                counter += 1
    return file_list

# path to the datasets
'''root_dir = 'Dataset/training'
filenames = sorted(get_file_list(root_dir))'''
'''We now define a variable that will store all of the features, go through all filenames in the dataset, 
extract their features, and append them to the previously defined variable:'''

'''feature_list = []
for i in tqdm(range(len(filenames))):
    feature_list.append(extract_features(filenames[i], model))

pickle.dump(feature_list, open('features-caltech101-resnet.pickle', 'wb'))
pickle.dump(filenames, open('filenames-caltech101.pickle','wb'))
'''
