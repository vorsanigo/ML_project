import os
import glob
import numpy as np
import skimage.io
from skimage.transform import resize
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
        data_classes = os.listdir(data_path)  # subfolders -> classes (1, 2, distractor ...)

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
        images_names = []  # names of images

        img_length = self.img_length
        img_height = self.img_height

        for img_path in data_mapping.keys():

            if img_path.endswith('.jpg'):
                images_paths.append(img_path)
                temp = "r"+img_path
                images_names.append(os.path.split(temp)[1])

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

        return images_names, images_paths, images_arrays, np.array(classes)


# Read images with common extensions from a directory with no subfolders
def read_imgs_no_subfolders(dirPath, extensions=None):
    if extensions is None:
        extensions = ['.jpg', '.png', '.jpeg']

    all_img = []
    img_list = glob.glob(os.path.join(dirPath, '*'))
    # Iterate over images
    for ext in extensions:
        for img_path in img_list:
            if img_path.endswith(ext):
                new_img = skimage.io.imread(img_path, as_gray=False)
                new_img = resize(new_img, (100, 100), anti_aliasing=True, preserve_range=True)

                if new_img.shape[2] == 1:
                    new_img = np.repeat(new_img, 3, axis=2)
                if new_img.shape[2] == 4:
                    new_img = new_img[1:, :, :3]

                all_img.append(new_img)

    return all_img, img_list
