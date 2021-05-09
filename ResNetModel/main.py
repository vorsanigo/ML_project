import loader
import model
import feature_extractor
import pickle

loader = loader.Loader(224, 224, 3)
model_manager = model.ResNetPlus()
feature_extractor = feature_extractor.FeatureExtractor()



# TODO loading training data
x = loader.get_files('Dataset_1/training')
data_paths_train = loader.get_data_paths(x)
list_images_train = data_paths_train[1]
images_paths_train = data_paths_train[0]
print(images_paths_train)

#pickle.dump(images_paths, open('images_paths_gallery.pickle', 'wb'))

#pickle.dump(list_images, open('list_images_provae.pickle','wb'), protocol = 0)
#serialized = pickle.dumps(list_images, protocol=0) # protocol 0 is printable ASCII

# TODO create model ResNetPlus, train, predict features
model_res_net = model_manager.create_model()
y = model_manager.compile_train(model_res_net)
#x = model_res_net.predict(list_images_train[0])

# TODO loading gallery data
x = loader.get_files('Dataset_1/validation/gallery')
data_paths_gellery = loader.get_data_paths(x)
list_images_gellery = data_paths_gellery[1]
images_paths_gellery = data_paths_gellery[0]
print(images_paths_gellery)
# TODO predict features in gallery
single_features = feature_extractor.extract_features_single_img(list_images_gellery[0], model_res_net)
tot_features = feature_extractor.extract_tot_features(list_images_gellery, model_res_net)
print(list_images_gellery)
print('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
print(tot_features)

# TODO predict features query
x = loader.get_files('Dataset_1/validation/query/2/ec50k_00020003.jpg')
print("RRRRRRRRRRRRR")
data_paths_query = loader.get_data_paths(x)
list_images_query = data_paths_query[1]
images_paths_query = data_paths_query[0]
print(images_paths_query)
print(list_images_query)
# TODO predict features of query
features_query = feature_extractor.extract_tot_features(list_images_query, model_res_net)
print(features_query)



#pickle.dump(tot_features, open('list_tot_features_gallery.pickle', 'wb'))
# TODO use model ResNet normal
'''mod2 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
features = mod2.predict(list_images[0])
print(features)
flattened_features = features.flatten()
print(flattened_features)'''

# TODO -> we can see we have different features in different models
