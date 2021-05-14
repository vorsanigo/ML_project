import loader
import model
import feature_extractor
import pickle
from scipy import spatial
import numpy as np
import os
import argparse
import wandb
from wandb.keras import WandbCallback
import keras


# link seguito come spunto https://www.pyimagesearch.com/2020/04/27/fine-tuning-resnet-with-keras-tensorflow-and-deep-learning/

parser = argparse.ArgumentParser(description='Challenge presentation example')
parser.add_argument('--data_path',
                    '-d',
                    type=str,
                    default='challenge_data_small',
                    help='Dataset path')
parser.add_argument('-mode',
                    type=str,
                    default='training',
                    help='training or test')
parser.add_argument('-n',
                    type=str,
                    default='test')
parser.add_argument('-lr',
                    type=float,
                    default=1e-4)
parser.add_argument('-e',
                    type=int,
                    default=30)
parser.add_argument('-bs',
                    type=int,
                    default=32)
parser.add_argument('-wandb',
                    type=str,
                    default='True',
                    help='Log on WandB (default = True)')
parser.add_argument('-img_size',
                    type=int,
                    default=224)
parser.add_argument('-channels',
                    type=int,
                    default=3)
'''parser.add_argument('--descriptor',
                    '-desc',
                    type=str,
                    default='sift',
                    help='Descriptor to be used')
parser.add_argument('--output_dim',
                    '-o',
                    type=int,
                    default=10,
                    help='Descriptor length')
parser.add_argument('--save_dir',
                    '-s',
                    type=str,
                    default=None,
                    help='Save or not gallery/query feats')
parser.add_argument('--gray',
                    '-g',
                    action='store_true',
                    help='Grayscale/RGB SIFT')
parser.add_argument('--random',
                    '-r',
                    action='store_true',
                    help='Random run')'''
args = parser.parse_args()

'''wandb.login()

# trigger or untrigger WandB
if args.wandb == 'False' or args.mode == 'deploy':
    os.environ['WANDB_MODE'] = 'dryrun'

# 1. Start a W&B run
wandb.init(project='aml-challenge', entity='innominati', group=args.mode, name=args.n)
wandb.config.epochs = args.e
wandb.config.batch_size = args.bs

wandb.init(project='aml-challenge',
           config={  # and include hyperparameters and metadata
               "learning_rate": args.lr,
               "epochs": args.e,
               "batch_size": args.bs,
         })
config = wandb.config'''


# we define training dataset
training_path = os.path.join(args.data_path, 'training')

# we define validation dataset
validation_path = os.path.join(args.data_path, 'validation')
gallery_path = os.path.join(validation_path, 'gallery')
query_path = os.path.join(validation_path, 'query')


print("EP", args.e)

loader = loader.Loader(args.img_size, args.img_size, args.channels) # img_length, img_height, num_of_channels
model_manager = model.ResNetPlus(training_path, args.lr, args.bs, args.e) # train_path, lr_rate, batch_size, num_epochs
feature_extractor = feature_extractor.FeatureExtractor()

#loaded_model_resnet = keras.models.load_model("resnet_model")


'''training_path = 'Dataset_1/training'
gallery_path = 'Dataset_1/validation/gallery'
query_path = 'Dataset_1/validation/query'
'''



# TODO loading training data
x = loader.get_files(training_path)
images_paths_train, list_images_train, images_classes_train = loader.get_data_paths(x)
#print("paths images train", images_paths_train)

#pickle.dump(images_paths, open('images_paths_gallery.pickle', 'wb'))

#pickle.dump(list_images, open('list_images_provae.pickle','wb'), protocol = 0)


# TODO create model ResNetPlus, train it
model_res_net = model_manager.create_model()
y = model_manager.compile_train(model_res_net)
#y = model_manager.compile_train_wandb(model_res_net)
# TODO save the model


# TODO loading gallery data
x = loader.get_files(gallery_path)
images_paths_gallery, list_images_gallery, images_classes_gallery = loader.get_data_paths(x)
#print("paths images paths gallery", images_paths_gallery)
# TODO predict features in gallery
#single_features = feature_extractor.extract_features_single_img(list_images_gallery, model_res_net)
features_gallery = feature_extractor.extract_tot_features(list_images_gallery, model_res_net)
'''print(list_images_gallery)
print('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
print("features gallery", tot_features_gallery)'''
'''pickle.dump(tot_features_gallery, open('features_gallery_tot.pickle', 'wb'))
pickle.dump(images_classes_gallery, open('gallery_classes_tot.pickle', 'wb'))'''

# TODO loading query data
x = loader.get_files(query_path)
#print("RRRRRRRRRRRRR")
images_paths_query, list_images_query, images_classes_query = loader.get_data_paths(x)
# images_paths_query, list_images_query, images_classes = loader.get_data_paths(x)
'''print(images_paths_query)
print(list_images_query)'''
# TODO predict features of query
features_query = feature_extractor.extract_tot_features(list_images_query, model_res_net)
#print("features query", features_query)
'''pickle.dump(features_query, open('features_query_tot.pickle', 'wb'))
pickle.dump(images_classes_query, open('query_classes_tot.pickle', 'wb'))'''


'''gallery_features = pickle.load(open('features_gallery_tot.pickle', 'rb'))
query_features = pickle.load(open('features_query_tot.pickle', 'rb'))
gallery_classes = pickle.load(open('gallery_classes_tot.pickle', 'rb'))
query_classes = pickle.load(open('query_classes_tot.pickle', 'rb'))'''

print(features_gallery)
print(len(features_gallery))
print(features_query)
print(len(features_query))
print(images_classes_gallery)
print(images_classes_query)

# define the distance between query - gallery features vectors
pairwise_dist = spatial.distance.cdist(features_query, features_gallery, 'minkowski', p=2.)
# rows -> queries | columns -> gallery --> cell = distance between query-gallery image
print(pairwise_dist)
print(len(pairwise_dist))


print('--> Computed distances and got c-dist {}'.format(pairwise_dist.shape))

indices = np.argsort(pairwise_dist, axis=-1) # it associates to each distance its ranking, from the closest to the more dis
print(indices)


print('eeeeeeeeeeeeeeeeeee')
print("classes", images_classes_gallery)
gallery_matches = images_classes_gallery[indices]
print("matches", gallery_matches)


def topk_accuracy(gt_label, matched_label, k=1):
    matched_label = matched_label[:, :k]
    total = matched_label.shape[0]
    correct = 0
    for q_idx, q_lbl in enumerate(gt_label):
        correct+= np.any(q_lbl == matched_label[q_idx, :]).item()
    acc_tmp = correct/total

    return acc_tmp


print('########## RESULTS ##########')

for k in [1, 3, 10]:
    topk_acc = topk_accuracy(images_classes_query, gallery_matches, k)
    print('--> Top-{:d} Accuracy: {:.3f}'.format(k, topk_acc))

#pickle.dump(tot_features, open('list_tot_features_gallery.pickle', 'wb'))
# TODO use model ResNet normal
'''mod2 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
features = mod2.predict(list_images[0])
print(features)
flattened_features = features.flatten()
print(flattened_features)'''

# TODO -> we can see we have different features in different models