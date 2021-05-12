import os
import pickle

import numpy as np
#from PIL import Image
#from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
#from pyimagesearch import config
#from imutils import paths
import random
import model

# TODO UNDERSTAND HOW TO SERIALIZE IMAGES IF POSSIBLE AFTER LOADING

class Loader():

    def __init__(self, img_length, img_height, num_channels):
        self.img_length = img_length
        self.img_height = img_height
        self.num_channels = num_channels


    def get_files(self, data_path):
        '''
        Given the path of a dataset, it returns dictionary with keys the paths of single images, values the corresponding
        class
        :param data_path:
        :return:
        '''
        assert os.path.exists(data_path), 'Insert a valid path!'

        data_classes = os.listdir(data_path)

        data_mapping = {}

        for c, c_name in enumerate(data_classes):
            temp_path = os.path.join(data_path, c_name)
            temp_images = os.listdir(temp_path)

            for i in temp_images:
                img_tmp = os.path.join(temp_path, i)

                if img_tmp.endswith('.jpg'):
                    if c_name == 'distractor':
                        data_mapping[img_tmp] = -1
                    else:
                        data_mapping[img_tmp] = int(c_name)

        print('Loaded {:d} from {:s} images'.format(len(data_mapping.keys()), data_path))

        return(data_mapping)


    def get_data_paths(self, data_mapping):
        '''
        Given a dictionary from the previous function, it returns one tuple with one list of images paths, one of images
        arrays, one np.array of the classes
        :param data_mapping:
        :return:
        '''
        images_paths = []
        images_arrays = []
        classes = []

        img_length = self.img_length
        img_height = self.img_height

        for img_path in data_mapping.keys():

            if img_path.endswith('.jpg'):
                images_paths.append(img_path)

                '''img = Image.open(img_path)
                img = np.asarray(img)
                img = np.resize(img, (img_size, img_size))
                images_arrays.append(img)'''

                img = load_img(img_path, target_size=(img_length, img_height)) # img_length = img_height = 224
                img = img_to_array(img)
                print(img)
                print(img.shape) # (224, 224, 3)

                #img1 = np.random.rand(224, 224, 4)

                # reshape dimension of channels to 3
                if img.shape[2] == 1:
                    img = np.repeat(img, 3, axis=2)
                if img.shape[2] == 4:
                    img = img[1:, :, :3]



                # preprocess the image by (1) expanding the dimensions and
                # (2) subtracting the mean RGB pixel intensity from the
                # ImageNet dataset
                img = np.expand_dims(img, axis=0) # it adds first component -> (1, 224, 224, 3)
                print("SSS", img.shape)
                img = preprocess_input(img)
                images_arrays.append(img)
                print(img.shape)
                '''mod = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
                features = mod.predict(img)
                flattened_features = features.flatten()
                print(flattened_features)'''

                classes.append(data_mapping[img_path])

        images_arrays = np.array(images_arrays) / 255.0 # put -0.5 ???
        print(type(images_arrays))
        #images_arrays.reshape(-1, img_length, img_height, 1)
        #images_arrays.reshape(img_length, img_height, 1)
        print(type(images_arrays))
        print(images_arrays[0].shape)

        '''input_shape = (224, 224, 3)
        img = image.load_img(img_path, target_size=(
            input_shape[0], input_shape[1]))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        features = model.predict(preprocessed_img)
        flattened_features = features.flatten()
        normalized_features = flattened_features / norm(flattened_features)'''

        return images_paths, images_arrays, np.array(classes)

    def num_classes(self, data_classes):
        '''
        It returns the number of classes
        :param data_classes:
        :return:
        '''
        return len(data_classes)



'''
# TODO loading data
loader = Loader(224, 224, 3)
x = loader.get_files('Dataset_1/training')
print(x)
e = loader.get_data_paths(x)
print(e[1].shape)
list_images = e[1]
print(e[1][1].shape)
#pickle.dump(list_images, open('list_images_provae.pickle','wb'), protocol = 0)
#serialized = pickle.dumps(list_images, protocol=0) # protocol 0 is printable ASCII

# TODO create model ResNetPlus, train, predict features
model = model.ResNetPlus()
mod = model.create_model()
mod.summary()
y = model.compile_train(mod)
x = mod.predict(list_images[0])
print('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
print(mod.predict(list_images[0]))
print(x.shape)'''

# TODO use model ResNet normal
'''mod2 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
features = mod2.predict(list_images[0])
print(features)
flattened_features = features.flatten()
print(flattened_features)
'''
# TODO -> we can see we have different features in different models


# shape of images_arrays -> (131, 1, 224, 224, 3)
# shape of single image -> (1, 224, 224, 3)
'''
list_images = pickle.load(open('/home/veror/Desktop/Uni/Applied Machine Learning/prova resnet/list_images_keras.pickle', 'rb'))
print(list_images[0].shape)


mod = model.crate_model()
mod.summary()

x = mod.predict(list_images[0])
print(mod.predict(list_images[0]))
print(x.shape)'''

'''img_size = 224

extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']'''


'''def get_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)'''

'''
def get_file_list(root_dir):
    
    Given the path of a single dataset (eg: train, gallery, query, test) it returns a list containing all the files
    in the directory
    :param root_dir:
    :return: list of files in a single dataset
    
    file_list = []
    counter = 1
    for root, directories, filenames in os.walk(root_dir):

        print(directories)
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                file_list.append(os.path.join(root, filename))
                counter += 1
    return file_list'''


'''
def preprocess_image(file_list):
#Normalize data and reshape them
#:param file_list:
#:return: normalized and reshaped list of data

input_shape = (img_size, img_size, 3)
img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
features = model.predict(preprocessed_img)
flattened_features = features.flatten()
normalized_features = flattened_features / norm(flattened_features)
return normalized_features
# normalize data
array_list = np.array(file_list) / 255.0
# reshape
array_list.reshape(-1, img_size, img_size, 1)
return array_list'''


'''x = get_file_list('Dataset/training')
for i in x:
    print(x)'''
'''
y = preprocess_image(x)
for e in y:
    print(e)
'''





