import loader
import model
import feature_extractor
import pickle
from scipy import spatial
import numpy as np
import os
import argparse



parser = argparse.ArgumentParser(description='Challenge presentation example')
parser.add_argument('--data_path',
                    '-d',
                    type=str,
                    default='challenge_data_small',
                    help='Dataset path')
parser.add_argument('--descriptor',
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
                    help='Random run')
args = parser.parse_args()


# we define training dataset
training_path = os.path.join(args.data_path, 'training')

# we define validation dataset
validation_path = os.path.join(args.data_path, 'validation')
gallery_path = os.path.join(validation_path, 'gallery')
query_path = os.path.join(validation_path, 'query')




loader = loader.Loader(224, 224, 3)
model_manager = model.ResNetPlus()
feature_extractor = feature_extractor.FeatureExtractor()

'''training_path = 'Dataset_1/training'
gallery_path = 'Dataset_1/validation/gallery'
query_path = 'Dataset_1/validation/query''''



# TODO loading training data
x = loader.get_files(training_path)
data_paths_train = loader.get_data_paths(x)
list_images_train = data_paths_train[1]
images_paths_train = data_paths_train[0]
print("paths images train", images_paths_train)

#pickle.dump(images_paths, open('images_paths_gallery.pickle', 'wb'))

#pickle.dump(list_images, open('list_images_provae.pickle','wb'), protocol = 0)
#serialized = pickle.dumps(list_images, protocol=0) # protocol 0 is printable ASCII

# TODO create model ResNetPlus, train, predict features
model_res_net = model_manager.create_model()
y = model_manager.compile_train(model_res_net)
#x = model_res_net.predict(list_images_train[0])

# TODO loading gallery data
x = loader.get_files(gallery_path)
data_paths_gellery = loader.get_data_paths(x)
list_images_gellery = data_paths_gellery[1]
images_paths_gellery = data_paths_gellery[0]
print("paths images paths gallery", images_paths_gellery)
gallery_classes = data_paths_gellery[2]
print("gallery classes", gallery_classes)
# TODO predict features in gallery
single_features = feature_extractor.extract_features_single_img(list_images_gellery[0], model_res_net)
tot_features = feature_extractor.extract_tot_features(list_images_gellery, model_res_net)
print(list_images_gellery)
print('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
print("features gallery", tot_features)
pickle.dump(tot_features, open('features_gallery.pickle', 'wb'))
pickle.dump(gallery_classes, open('gallery_classes.pickle', 'wb'))

# TODO predict features query
x = loader.get_files(query_path)
print("RRRRRRRRRRRRR")
data_paths_query = loader.get_data_paths(x)
list_images_query = data_paths_query[1]
images_paths_query = data_paths_query[0]
query_classes = data_paths_query[2]
# images_paths_query, list_images_query, images_classes = loader.get_data_paths(x)
print(images_paths_query)
print(list_images_query)
# TODO predict features of query
features_query = feature_extractor.extract_tot_features(list_images_query, model_res_net)
print("features query", features_query)
pickle.dump(features_query, open('features_query.pickle', 'wb'))
pickle.dump(query_classes, open('query_classes.pickle', 'wb'))


gallery_features = pickle.load(open('features_gallery.pickle', 'rb'))
query_features = pickle.load(open('features_query.pickle', 'rb'))
gallery_classes = pickle.load(open('gallery_classes.pickle', 'rb'))
query_classes = pickle.load(open('query_classes.pickle', 'rb'))

print(gallery_features)
print(len(gallery_features))
print(query_features)
print(len(query_features))
print(gallery_classes)
print(query_classes)


pairwise_dist = spatial.distance.cdist(query_features, gallery_features, 'minkowski', p=2.)
print(pairwise_dist)
print(len(pairwise_dist))


print('--> Computed distances and got c-dist {}'.format(pairwise_dist.shape))

indices = np.argsort(pairwise_dist, axis=-1) # it associates to each distance its ranking, from the closest to the more dis
print(indices)


print('eeeeeeeeeeeeeeeeeee')
gallery_matches = gallery_classes[indices]
print(gallery_matches)


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
    topk_acc = topk_accuracy(query_classes, gallery_matches, k)
    print('--> Top-{:d} Accuracy: {:.3f}'.format(k, topk_acc))

#pickle.dump(tot_features, open('list_tot_features_gallery.pickle', 'wb'))
# TODO use model ResNet normal
'''mod2 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
features = mod2.predict(list_images[0])
print(features)
flattened_features = features.flatten()
print(flattened_features)'''

# TODO -> we can see we have different features in different models
