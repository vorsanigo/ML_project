# todo rendere size no hard code
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize


def normalize_img(imgs):

    """ Normalize images """

    transformed_images = [img/255 for img in imgs]
    return transformed_images


def random_crop(img):

    """" Crop images """

    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]  # for us it is the one set in the loader (eg: 324)
    dy, dx = 224, 224  # img will be reduced at most to this size (eg: 224)
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return resize(img[y:(y+dy), x:(x+dx), :], (324, 324, 3), anti_aliasing=True, preserve_range=True)


def data_augmentation(train_set, batch_size):

    """ Data augmentation on images of the dataset """

    # Initialize the training training data augmentation object
    trainAug = ImageDataGenerator(
        rotation_range=25,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
        preprocessing_function=random_crop)

    trainAug.fit(train_set) # TODO serve ?

    trainGen = trainAug.flow(
        train_set,
        train_set,
        shuffle=True,
        batch_size=batch_size)

    """
    # code to see the modified images
    for _ in range(10):
        img, label = trainGen.next()
        print(img.shape)  # (1,256,256,3)
        plt.imshow(img[0])
        plt.show()"""

    return trainGen




def data_augmentation_triplet(set_anchor, set_positive, set_negative, batch_size):

    ''' Data augmentation on images of the dataset for triplets '''

    # initialize the training training data augmentation object
    trainAug = ImageDataGenerator(
        rotation_range=25,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        #fill_mode="nearest",
        preprocessing_function=random_crop
    )

    #trainAug.fit(train_set) # TODO serve ?

    local_seed = 10

    trainGenAnchor = trainAug.flow(
        set_anchor,
        set_anchor,
        seed=local_seed,
        #steps_per_epoch=13//batch_size,
        shuffle=False,
        batch_size=batch_size
    )

    trainGenPositive = trainAug.flow(
        set_positive,
        set_positive,
        seed=local_seed,
        #steps_per_epoch=13//batch_size,
        shuffle=False,
        batch_size=batch_size
    )

    trainGenNegative = trainAug.flow(
        set_negative,
        set_negative,
        seed=local_seed,
        #steps_per_epoch=13//batch_size,
        shuffle=False,
        batch_size=batch_size
    )

    # code to see the modified images
    '''for _ in range(10):
    
        img, label = trainGenAnchor.next()
        print(img.shape)  # (1,256,256,3)
        plt.imshow(img[0])
        plt.show()
    
        img, label = trainGenPositive.next()
        print(img.shape)  # (1,256,256,3)
        plt.imshow(img[0])
        plt.show()
    
        img, label = trainGenNegative.next()
        print(img.shape)  # (1,256,256,3)
        plt.imshow(img[0])
        plt.show()'''

    return trainGenAnchor, trainGenPositive, trainGenNegative
