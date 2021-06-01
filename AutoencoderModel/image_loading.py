import os
import glob
import numpy as np
import skimage.io
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


class Loader:
    
    """This class reads images with common extensions from a directory for later use"""

    def __init__(self, img_length, img_height, num_channels):
        
        self.img_length = img_length
        self.img_height = img_height
        self.num_channels = num_channels

    def get_files(self, data_path):

        """This function returns the data mapping"""
        
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
                        if c_name.isdigit():
                            data_mapping[img_tmp] = int(c_name)
                        else:
                            data_mapping[img_tmp] = c_name
                elif img_tmp.endswith('.jpeg'):
                    if c_name == 'distractor':
                        data_mapping[img_tmp] = -1
                    else:
                        if c_name.isdigit():
                            data_mapping[img_tmp] = int(c_name)
                        else:
                            data_mapping[img_tmp] = c_name
                elif img_tmp.endswith('.png'):
                    if c_name == 'distractor':
                        data_mapping[img_tmp] = -1
                    else:
                        if c_name.isdigit():
                            data_mapping[img_tmp] = int(c_name)
                        else:
                            data_mapping[img_tmp] = c_name


        print('\nLoaded {:d} from {:s} images'.format(len(data_mapping.keys()), data_path))

        return data_mapping

    def get_data_paths(self, data_mapping):

        """This function returns the image path, the image name and the image as a numpy array"""

        images_paths = []  
        images_arrays = [] 
        classes = []  
        images_names = []  

        img_length = self.img_length
        img_height = self.img_height

        print('\nProcessing images...')

        for img_path in data_mapping.keys():

            if img_path.endswith('.jpg'):
                images_paths.append(img_path)
                temp = "r"+img_path
                images_names.append(os.path.split(temp)[1])

                img = load_img(img_path, target_size=(img_length, img_height))

                img = img_to_array(img)

                if img.shape[2] == 1:
                    img = np.repeat(img, 3, axis=2)
                if img.shape[2] == 4:
                    img = img[:, :, :3]

                images_arrays.append(img)
                classes.append(data_mapping[img_path])

            elif img_path.endswith('.jpeg'):
                images_paths.append(img_path)
                temp = "r"+img_path
                images_names.append(os.path.split(temp)[1])

                img = load_img(img_path, target_size=(img_length, img_height))

                img = img_to_array(img)

                if img.shape[2] == 1:
                    img = np.repeat(img, 3, axis=2)
                if img.shape[2] == 4:
                    img = img[:, :, :3]

                images_arrays.append(img)
                classes.append(data_mapping[img_path])

            elif img_path.endswith('.png'):
                images_paths.append(img_path)
                temp = "r"+img_path
                images_names.append(os.path.split(temp)[1])

                img = load_img(img_path, target_size=(img_length, img_height))

                img = img_to_array(img)

                if img.shape[2] == 1:
                    img = np.repeat(img, 3, axis=2)
                if img.shape[2] == 4:
                    img = img[:, :, :3]

                images_arrays.append(img)
                classes.append(data_mapping[img_path])

        print('\nImages processed')

        return images_names, images_paths, images_arrays, np.array(classes)


def read_imgs_no_subfolders(dirPath, img_size, extensions=None):

    """This function reads images with common extensions from a directory with no subfolders"""

    if extensions is None:
        extensions = ['.jpg', '.png', '.jpeg']

    all_img = []
    img_list = glob.glob(os.path.join(dirPath, '*'))

    for ext in extensions:
        for img_path in img_list:
            if img_path.endswith(ext):

                img = load_img(img_path, target_size=(img_size, img_size))

                img = img_to_array(img)

                if img.shape[2] == 1:
                    img = np.repeat(img, 3, axis=2)
                if img.shape[2] == 4:
                    img = img[:, :, :3]

                all_img.append(img)

    return all_img, img_list

