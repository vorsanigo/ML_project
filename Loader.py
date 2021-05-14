import os

from cv2 import cv2
import numpy as np




class Dataset(object):
    def __init__(self, data_path):
        self.data_path = data_path
        assert os.path.exists(self.data_path), 'Insert a valid path!'

        self.data_classes = os.listdir(self.data_path)

        self.data_mapping = {}

        width = 128
        height = 128
        n_color_channels = 3

        for c, c_name in enumerate(self.data_classes):
            temp_path = os.path.join(self.data_path, c_name)
            temp_images = os.listdir(temp_path)

            for i in temp_images:
                img_tmp = os.path.join(temp_path, i)

                if img_tmp.endswith('.jpg'):

                    # read and resize the image
                    img_array = (cv2.imread(img_tmp) / 255.0) - 0.5 # normalization
                    img = cv2.resize(img_array, (width, height), interpolation=cv2.INTER_CUBIC)

                    if c_name == 'distractor':
                        self.data_mapping[img_tmp] = (img, -1) # tuple containing the array image resized and the label
                    else:
                        self.data_mapping[img_tmp] = (img, int(c_name))

        print('Loaded {:d} from {:s} images'.format(len(self.data_mapping.keys()),
                                                    self.data_path))

    def get_data_paths(self):
        images_path = []
        classes = []
        images_array = []
        for img_path in self.data_mapping.keys():
            if img_path.endswith('.jpg'):
                images_path.append(img_path)
                classes.append(self.data_mapping[img_path][1])
                images_array.append(self.data_mapping[img_path][0])
        return images_path, np.array(classes), images_array


    def num_classes(self):
        return len(self.data_classes)


'''class Loader:

    def load_data(self, data_path, ):'''