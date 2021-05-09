import loader
import model
import feature_extractor

loader = loader.Loader(224, 224, 3)
model_manager = model.ResNetPlus()
feature_extractor = feature_extractor.FeatureExtractor()



# TODO loading data
x = loader.get_files('Dataset_1/training')
print(x)
data_paths = loader.get_data_paths(x)
print(data_paths[1].shape)
list_images = data_paths[1]
print(data_paths[1][1].shape)

#pickle.dump(list_images, open('list_images_provae.pickle','wb'), protocol = 0)
#serialized = pickle.dumps(list_images, protocol=0) # protocol 0 is printable ASCII

# TODO create model ResNetPlus, train, predict features
model_res_net = model_manager.create_model()
y = model_manager.compile_train(model_res_net)
x = model_res_net.predict(list_images[0])
'''tot_features = feature_extractor.extract_features(list_images, model_res_net)
print(list_images)
print(list_images[0])
print(list_images)
print('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
print(model_res_net.predict(list_images[0]))
print(x.shape)
print(tot_features)'''

# TODO use model ResNet normal
'''mod2 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
features = mod2.predict(list_images[0])
print(features)
flattened_features = features.flatten()
print(flattened_features)'''

# TODO -> we can see we have different features in different models
