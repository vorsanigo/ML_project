import os
import glob
import random

import out as out
from PIL import Image, ImageEnhance, ImageFilter
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def normalize_img(imgs):
    transformed_images = [img/255 for img in imgs]
    return transformed_images

# TODO cropping maybe to add
def data_augmentation(train_set, batch_size):
    '''
    Data augmentation on images of the dataset
    :return:
    '''

    # initialize the training training data augmentation object
    trainAug = ImageDataGenerator(
        rotation_range=25,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest")


    # initialize the training generator -> from directory -> not used now
    '''trainGen = trainAug.flow_from_directory(
        dir_path,
        class_mode="categorical",
        target_size=target_size,
        color_mode="rgb",
        shuffle=True,# TODO true or false ? true è default a per noi è ok, anche se ci scambia l'ordine non ci interessa se non usiamo le classe, se usiamo le classi boh
        batch_size=batch_size)'''

    trainAug.fit(train_set) # TODO serve ?

    trainGen = trainAug.flow(
        train_set,
        train_set,
        #class_mode="categorical",
        #target_size=target_size,
        #color_mode="rgb",
        shuffle=True,
        batch_size=batch_size

        # to save modified images
        #save_to_dir = "output/",
        #save_prefix = "",
        #save_format = 'png',
    )

    # code to see the modified images
    '''
    for _ in range(10):
        img, label = trainGen.next()
        print(img.shape)  # (1,256,256,3)
        plt.imshow(img[0])
        plt.show()'''

    return trainGen




'''datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale = 1. / 255,
    rotation_range=25,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

dir_It = datagen.flow_from_directory(
    "dataset/training",
    batch_size=1,
    save_to_dir="output/",
    save_prefix="",
    save_format='png',
)

for _ in range(10):
    img, label = dir_It.next()
    print(img.shape)  # (1,256,256,3)
    plt.imshow(img[0])
    plt.show()'''


'''
def modifyImage(data_path):

    # Iterate over subfolders
    for image_path in glob.glob(os.path.join(data_path, '*')):
        dir = os.chdir(image_path)
        temp = 0
        # Getting list of all images in a subfolder
        filelist = glob.glob(os.path.join(image_path, '*.jpg'))

        # Iterate over images
        for j, path in enumerate(filelist):
            if j < 2:
                transform_list = [1,2,3,4]
                # Getting random transformation
                transf = random.choice(transform_list)

                # Converting into "Image" format
                originalImage = Image.open(path)
                img = originalImage.convert('RGB')

                if temp != transf:
                    if transf == 1:
                        blurring(img, dir, j)
                    elif transf == 2:
                        cropping(img, dir, j)
                    elif transf == 3:
                        rotate(img, dir, j)
                    else:
                        light(img, dir, j)
                    temp = transf


def blurring(img, image_path, j):
    blur = img.filter(ImageFilter.BoxBlur(5))

    blur.save("blur_%s.jpg" % (j))


def cropping(img, image_path, j):

    width, height = img.size
    left = (180, 1, width, height)
    top = (1, 180, width, height)
    right = (1, 1, 4.2 * width / 6, height)
    bottom = (1, 1, width, 4.2 * height / 6)
    alldim = (100, 100, 4.2 * width / 6, 4.2 * height / 6)

    cropl = img.crop(left)
    cropup = img.crop(top)
    cropr = img.crop(right)
    cropdown = img.crop(bottom)
    cropall = img.crop(alldim)

    cropl.save("cropl_%s.jpg" % (j))
    cropup.save("cropup_%s.jpg" % (j))
    cropr.save("cropr_%s.jpg" % (j))
    cropdown.save("cropdown_%s.jpg" % (j))
    cropall.save("cropall_%s.jpg" % (j))


def rotate(img, image_path, j):

    rotatedr = img.rotate(45)
    rotatedl = img.rotate(270)
    transposed = img.transpose(Image.ROTATE_90)
    flipped = img.transpose(Image.ROTATE_180)

    rotatedr.save("rotater_%s.jpg" % (j))
    transposed.save("transp_%s.jpg" % (j))
    flipped.save("flip_%s.jpg" % (j))
    rotatedl.save("rotatel_%s.jpg" % (j))


def light(img, image_path, j):

    # Brightness, contrast, saturation
    fil1 = ImageEnhance.Brightness(img)
    bright = fil1.enhance(1.3)
    fil2 = ImageEnhance.Contrast(img)
    contr = fil2.enhance(4)
    fil3 = ImageEnhance.Color(img)
    sat = fil3.enhance(4)

    bright.save("bright_%s.jpg" % (j))
    contr.save("contr_%s.jpg" % (j))
    sat.save("sat_%s.jpg" % (j))
'''

