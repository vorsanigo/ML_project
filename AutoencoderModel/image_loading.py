import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


# Read images with common extensions from a directory
class Loader():

    def __init__(self, img_length, img_height, num_channels):
        self.img_length = img_length
        self.img_height = img_height
        self.num_channels = num_channels

    def get_files(self, data_path):
        assert os.path.exists(data_path), 'Insert a valid path!'
        data_classes = os.listdir(data_path) # subfolders -> classes (1, 2, distractor ...)

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

        print('\nLoaded {:d} from {:s} images'.format(len(data_mapping.keys()), data_path))

        return data_mapping

    def get_data_paths(self, data_mapping):
        images_paths = []  # images paths
        images_arrays = []  # images as arrays
        classes = []  # classes of images

        img_length = self.img_length
        img_height = self.img_height

        for img_path in data_mapping.keys():

            if img_path.endswith('.jpg'):
                images_paths.append(img_path)

                # load image with chosen size
                img = load_img(img_path, target_size=(img_length, img_height))

                # transform image into array
                img = img_to_array(img)

                # reshape dimension of channels to 3
                if img.shape[2] == 1:
                    img = np.repeat(img, 3, axis=2)
                if img.shape[2] == 4:
                    img = img[1:, :, :3]

                images_arrays.append(img)
                classes.append(data_mapping[img_path])

        return images_paths, images_arrays, np.array(classes)
