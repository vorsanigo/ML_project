# ML_project ~ Reverse Image Search 

Description. Let people know what your project can do specifically. ...
Badges. ...
Visuals. ...
Installation. ...
Usage. ...
Support. ...
Roadmap.

##Description

The purpose of this project is to present some possible machine learning solutions to a __reverse image search task__. 

Accordingly, we have implemented three main models: a 'simple' _autoencoder_, an _autoencoder_ implemented with a _triplet loss_ 
and a pretrained model, a _ResNet 50_.

All the labriries and packages used for the project are reported in the txt file _requiremenst.txt_ 

More specifically, the __main task__ was the following : if we give an input _query_ image to the algorithm, it 
should be able to match this input query image with other similar images taken from the _gallery_ data set. Therefore, 
the algorithm's final output is a ranked list of the matches between the query image and the ones in the gallery. 



## Structure


### Dataset files

However, as often happens in machine learning before moving to the main task it is essential to start by 
solving a __secondary__ but essential __task__: retrieve a reasonably good amount of data that will be used in our model. 
To do so we have created the following files: 

* `image_scarping.py` through which we download images directly from the google search engine by just passing a list of 
  places objects or whatever you may need.  

* `rename.py` & `create_the_queryset.py` are two additional files that have been used to change the name of the subfolders 
  where the different pictures where saved. Indeed, the name of the subfolders has to be numeric. On the other hand, with 
  the second file we just create our query gallery which is just a reduced copy of the first. 

* `transform.py` which is used for _data augmentation_. As a matter of fact, here different functions are implmented  to 
  normalize, randomly crop, rotate, zoom, shift and flip the images during the _training phase_. These transformations should 
  helpful to better train the _autoencoder_.

* `create_folders.py` is un additional file that has been used to create a different sub_folder for each subject in the 
  `ukbench`. This data set consists of 1000 images for 250 different objects (so, each objsect has 4 different images). 
  This data set has been retrived from the following link: (https://drive.google.com/file/d/0BwzOKB8koa9lR3pVTU1wMkJtamM/view)  


### Models

Going back to the main task, we now consider the core of the project:

* `autoencoder.py` In this file we have implemented both our autoencoders: the simple one and the one with the 
  _triplet loss_. In this file we have implemented functions for setting the _neural network architecture_, for creating, 
  saving, compiling and fitting the models the model. 

* `main_img_retrival.py`, `main_triplets.py` & `main_pretrained.py`These three files are the __essence__ of the project. 
  Each of them is associated to a model: respectively, to the autoencoder, the autoencoder with the triplet loss and 
  the ResNet50. By running these files you create the model and  __execute the reverse image task__. Before running the 
  code you can specify some parameters such as the batch size (e.g. -bs 32 -e 60) or the epochs thanks to the parser. By 
  running these pieces of code you can also visualize the output and see how the model performed.


### Utils

Lastly, there are few additional files:

* `visualization.py` which returns a nice graphical representation of the output: for each query image you have back the 
  query image itself plus the first ten closest images. 

* `final_display.py` which returns the output of the model in a dictionary structure where each query image is a key and 
  the associated value is a list with the names of the ten closest images. 

* `triplets.py` is an additional file used for the _autoencoder with the triplet loss_. In here, have been implemented 
  different functions to support the construction of this mdel.  
  

## Execution

### 1) Prerequistes

1) Installation of `Python 3.8.5` or more
2) It is recommended to create a virtual environment for the project
3) Installation of the required libraries through the `requirements.txt` file

### 2) Training

To train the autoencoder models (both with and without _triplet loss_):

1) **Dataset**: folder containing two subfolders called _train_ and _validation_, _validation_ must contain other two subfolders 
called _query_ end _gallery_ -> _train_, _query_, _gallery_ must contain images
2) Run the file with `name_file.py`: `main_img_retrival.py`or `main_triplets.py` or`main_pretrained.py` with the command`python name_file.py`,
some parameters can be added, as shown in the parser at the beginning of each file, the default ones are set
3) If `main_img_retrival.py` or `main_triplets.py` are run, the models are saved in the folder `output`

### 3) Test

It is possible to run all the models in _test_ mode by setting the parameter `-mode test` when running 
the three main files

### 4) Submission test

To test the models with also submitting the results to one server, it is necessary to have a dataset with the following 
structure: one folder _query_ and one _gallery_ containing the query and gallery images

Through the file `main.test.py` it is possible to test the models and to submit the results, three parameters can 
be added to respectvely test the _pretrained_ model, the _autoencoder_ without triplet loss, the _autoencoder with triplet 
loss_: `-model pretrained`, `- model convAE`, `-model triplets_loss`
