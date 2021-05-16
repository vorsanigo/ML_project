import os
#import args
from Loader import Dataset


'''parser = argparse.ArgumentParser(description='Challenge solution')
parser.add_argument('--data_path',
                    '-d',
                    type=str,
                    default='dataset',
                    help='Dataset path')
parser.add_argument('--output_dim',
                    '-o',
                    type=int,
                    default=20,
                    help='Descriptor length')
parser.add_argument('--save_dir',
                    '-s',
                    type=str,
                    default=None,
                    help='Save or not gallery/query feats')
parser.add_argument('--random',
                    '-r',
                    action='store_true',
                    help='Random run')
args = parser.parse_args()'''


def main():

    args_data_path = '../Dataset'
    # we define training dataset
    training_path = os.path.join(args_data_path, 'training')

    # we define validation dataset
    validation_path = os.path.join(args_data_path, 'validation')
    gallery_path = os.path.join(validation_path, 'gallery')
    query_path = os.path.join(validation_path, 'query')

    training_dataset = Dataset(data_path=training_path)
    gallery_dataset = Dataset(data_path=gallery_path)
    query_dataset = Dataset(data_path=query_path)

    # get training data and classes
    training_paths, training_classes, training_images = training_dataset.get_data_paths()

    # we get validation gallery and query data

    gallery_paths, gallery_classes, gallery_images = gallery_dataset.get_data_paths()
    query_paths, query_classes, query_images = query_dataset.get_data_paths()

    print(gallery_paths)
    print(gallery_classes)
    print(training_dataset)
    print(training_dataset[0])
    #if not args.random:


    '''feature_extractor = cv2.SIFT_create()

    # we define model for clustering
    model = KMeans(n_clusters=args.output_dim, n_init=10, max_iter=5000, verbose=False)
    # model = MiniBatchKMeans(n_clusters=args.output_dim, random_state=0, batch_size=100, max_iter=100, verbose=False)
    scale = StandardScaler()

    # we define the feature extractor providing the model
    extractor = FeatureExtractor(feature_extractor=feature_extractor,
                                 model=model,
                                 scale=scale,
                                 out_dim=args.output_dim)

    # we fit the KMeans clustering model
    extractor.fit_model(training_paths)

    extractor.fit_scaler(training_paths)
    # now we can use features
    # we get query features
    query_features = extractor.extract_features(query_paths)
    query_features = extractor.scale_features(query_features)

    # we get gallery features
    gallery_features = extractor.extract_features(gallery_paths)
    gallery_features = extractor.scale_features(gallery_features)

    print(gallery_features.shape, query_features.shape)


    pairwise_dist = spatial.distance.cdist(query_features, gallery_features, 'minkowski', p=2.)

    print('--> Computed distances and got c-dist {}'.format(pairwise_dist.shape))

    indices = np.argsort(pairwise_dist, axis=-1)

else:
    indices = np.random.randint(len(gallery_paths),
                                size=(len(query_paths) ,len(gallery_paths)))


gallery_matches = gallery_classes[indices]

print('########## RESULTS ##########')

for k in [1, 3, 10]:
    topk_acc = topk_accuracy(query_classes, gallery_matches, k)
    print('--> Top-{:d} Accuracy: {:.3f}'.format(k, topk_acc))'''



if __name__ == '__main__':
    main()