#import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib as plt
from wandb.keras import WandbCallback

import os
#from pyimagesearch import config
import numpy as np
import pickle
#from imutils import paths
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

# TODO change paths
# initialize the path to the *original* input directory of images
#BASE_PATH_DATASETS = "Dataset_1"
'''TRAIN_PATH = os.path.sep.join([BASE_PATH_DATASETS, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH_DATASETS, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH_DATASETS, "testing"])'''




#BASE_PATH_DATASETS = "Dataset_1"
'''TRAIN_PATH = "/content/drive/My Drive/Machine Learning Project/ML_challenge/ML_project/ResNetModel/dataset/training"
VAL_PATH = "/content/drive/My Drive/Machine Learning Project/ML_challenge/ML_project/ResNetModel/dataset/validation/gallery"
TEST_PATH = "/content/drive/My Drive/Machine Learning Project/ML_challenge/ML_project/ResNetModel/dataset/validation/query"
'''


# determine the total number of image paths in training, validation,
# and testing directories
totalTrain = 697 #len(list(paths.list_images(TRAIN_PATH)))
#totalVal = 15 #len(list(paths.list_images(VAL_PATH)))
#totalTest = len(list(paths.list_images(TEST_PATH)))

# TODO TAKE IT FROM loader.py
# define the names of the classes
#CLASSES = ["1", "9", "distractor"]

# initialize the initial learning rate, batch size, and number of
# epochs to train for
'''INIT_LR = 1e-4
BS = 32
NUM_EPOCHS = 3'''

# define the path to the serialized output model after training
#MODEL_PATH = "camo_detector.model"


class ResNetPlus():

    def __init__(self, train_path, init_lr, batch_size, num_epochs):
        self.train_path = train_path
        #self.val_path = val_path
        #self.test_path = test_path
        self.init_lr = init_lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def data_augmentation(self):
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
            fill_mode="nearest")
        # TODO DATA AUGMETATION SOLO SU TRAINING ??
        #valAug = ImageDataGenerator()
        # define the ImageNet mean subtraction (in RGB order) and set the
        # the mean subtraction value for each of the data augmentation
        # objects
        # TODO ???????? why mean?
        #mean = np.array([123.68, 116.779, 103.939], dtype="float32")
        #trainAug.mean = mean
        #valAug.mean = mean

        # TODO probably shuffle is already done here, so we do not have to do random shuffling at the beginning
        # initialize the training generator
        trainGen = trainAug.flow_from_directory(
            self.train_path,
            class_mode="categorical",
            target_size=(224, 224),
            color_mode="rgb",
            shuffle=True, # TODO true or false ? true è default a per noi è ok, anche se ci scambia l'ordine non ci interessa se non usiamo le classe, se usiamo le classi boh
            batch_size=self.batch_size)
        # initialize the validation generator
        '''valGen = valAug.flow_from_directory(
            VAL_PATH,
            class_mode="categorical",
            target_size=(224, 224),
            color_mode="rgb",
            shuffle=False,
            batch_size=BS)'''
        # initialize the testing generator
        '''testGen = valAug.flow_from_directory(
            TEST_PATH,
            class_mode="categorical",
            target_size=(224, 224),
            color_mode="rgb",
            shuffle=False,
            batch_size=BS)'''

        return trainGen#, valGen, testGen


    # Let’s load our ResNet50 classification model and prepare it for fine-tuning:
    # The process of fine-tuning allows us to reuse the filters learned during a previous training exercise. In our case,
    # we load ResNet50 pre-trained on the ImageNet dataset, leaving off the fully-connected (FC) head
    def create_model(self):
        '''
        Create the model starting from ResNet50 and appending more new layers
        :return:
        '''
        # initial weights set using imagenet
        baseModel = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        #baseModel.summary()
        # Here, we can observe that the final layer in the ResNet architecture (again, without the fully-connected layer head)
        # is an Activation layer that is 7 x 7 x 2048.

        # model construction
        headModel = baseModel.output
        headModel = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(headModel)
        headModel = tf.keras.layers.Flatten(name="flatten")(headModel)
        headModel = tf.keras.layers.Dense(256, activation="relu")(headModel)
        headModel = tf.keras.layers.Dropout(0.5)(headModel)
        #headModel = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(headModel)
        headModel = tf.keras.layers.Dense(22, activation="softmax")(headModel)

        # we append the HeadModel constructed to the body of ResNet
        model = Model(inputs=baseModel.input, outputs=headModel)
        model.summary()

        # loop over all layers in the base model and freeze them so they will *not* be updated during the training process
        for layer in baseModel.layers:
            layer.trainable = False

        return model


    def compile_train(self, model):
        '''
        Compile the model using data generator for augmentation
        :param model:
        :return:
        '''
        # set scheduler for learning rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.init_lr,
            decay_steps=250,
            decay_rate=0.96,
            staircase=True)
        # set optimizer
        opt = Adam(learning_rate=lr_schedule, decay=self.init_lr / self.num_epochs) #, decay=self.init_lr / self.num_epochs)
        #model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        # compile the model
        model.compile(loss="categorical_crossentropy", optimizer=opt)
        #model.compile(loss="mean_squared_error", optimizer=opt)

        # create random generators for data augmentation
        #generators = self.data_augmentation()
        trainGen = self.data_augmentation()
        #valGen = generators[1]

        '''print('s', type(trainGen))
        #print(valGen) # tuple
        print('q', trainGen[0]) # size -> num_of_images * 224 * 224 * 3
        print('e', trainGen[0][0][0].shape) # single image -> 224 * 224 * 3
        print(totalTrain)
        print(self.batch_size)
        print(self.num_epochs)'''

        # train the model
        print("[INFO] training model...")
        H = model.fit(
            trainGen,
            steps_per_epoch=totalTrain // self.batch_size,
            # validation_data=valGen,
            # validation_steps=1,#totalVal // BS,
            #batch_size=self.batch_size,
            epochs=self.num_epochs,
        )
        '''steps_per_epoch=totalTrain // self.batch_size,
        #batch_size=self.batch_size,
        epochs=3#self.num_epochs
        # validation_data=valGen,
        # validation_steps=1,#totalVal // BS,'''

        print("parameters")
        print(H.history.keys())

        '''# summarize history for accuracy
        plt.plot(H.history['accuracy'])
        plt.plot(H.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(H.history['loss'])
        plt.plot(H.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()'''


        model.save('resnet_model.h5')



        return H


    def compile_train_wandb(self, model):
        '''
        Compile the model using data generator for augmentation
        :param model:
        :return:
        '''
        # set scheduler for learning rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.init_lr,
            decay_steps=250,
            decay_rate=0.96,
            staircase=True)
        # set optimizer
        opt = Adam(learning_rate=lr_schedule, decay=self.init_lr / self.num_epochs) #, decay=self.init_lr / self.num_epochs)
        #model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        # compile the model
        model.compile(loss="categorical_crossentropy", optimizer=opt)
        #model.compile(loss="mean_squared_error", optimizer=opt)

        # create random generators for data augmentation
        #generators = self.data_augmentation()
        trainGen = self.data_augmentation()
        #valGen = generators[1]

        '''print('s', type(trainGen))
        #print(valGen) # tuple
        print('q', trainGen[0]) # size -> num_of_images * 224 * 224 * 3
        print('e', trainGen[0][0][0].shape) # single image -> 224 * 224 * 3
        print(totalTrain)
        print(self.batch_size)
        print(self.num_epochs)'''

        # train the model
        print("[INFO] training model...")
        H = model.fit(
            trainGen,
            steps_per_epoch=totalTrain // 32,
            # validation_data=valGen,
            # validation_steps=1,#totalVal // BS,
            #batch_size=self.batch_size,
            epochs=self.num_epochs,
            #callback=[WandbCallback()]
        )
        '''steps_per_epoch=totalTrain // self.batch_size,
        #batch_size=self.batch_size,
        epochs=3#self.num_epochs
        # validation_data=valGen,
        # validation_steps=1,#totalVal // BS,'''

        #wandb.log

        print("parameters")
        print(H.history.keys())

        # summarize history for accuracy
        plt.plot(H.history['accuracy'])
        plt.plot(H.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(H.history['loss'])
        plt.plot(H.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        model.save('resnet_model.h5')

        return H


# TODO TEST THE MODEL
'''res_mod = ResNetPlus()
mod = res_mod.create_model()
mod_fit = res_mod.compile_train(mod)


img = load_img('Dataset_1/training/1/ec50k_00010001.jpg', target_size=(224, 224))
plt.imshow(img)
plt.show()
img = img_to_array(img)
print(img.shape)
img = np.expand_dims(img, axis=0)
print(img.shape)
img = preprocess_input(img)
print(img.shape)

pred = mod.predict(img) / 255.0
print(pred)'''
# preprocess the image by (1) expanding the dimensions and
# (2) subtracting the mean RGB pixel intensity from the
# ImageNet dataset

# TODO END TEST MODEL

#list_im = pickle.load(open('list_images_model_res.pickle', 'rb'))



'''class ResNet(Model):

    def __init__(self): # input shape -> [244, 244, 3]

        super(ResNet, self).__init__(trainable=True)
        self.backbone = tf.keras.applications.ResNet50(include_top=False, input_shape=(244, 244, 3))
        self.final_conv = tf.keras.layers.Conv2D(filters=10, kernel_size=[3, 3], padding='valid')
        self.final_flatten = tf.keras.layers.Flatten()

    def call(self, inputs, training=None, mask=None):
        x = self.backbone(inputs)
        x = self.final_conv(x)
        x = self.final_flatten(x)
        return x, tf.keras.activations.softmax(x)'''
