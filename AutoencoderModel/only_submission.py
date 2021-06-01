from final_display import *
import pickle

file_knn_autoencoder = open('/dictionaries_submission_pickle/knn_autoencoder.pickle', 'rb')
knn_autoencoder = pickle.load(file_knn_autoencoder)
file_p_autoencoder = open('/dictionaries_submission_pickle/pairwise_autoencoder.pickle', 'rb')
p_autoencoder = pickle.load(file_p_autoencoder)

file_knn_pretrained = open('/dictionaries_submission_pickle/knn_pretrained.pickle', 'rb')
knn_pretrained = pickle.load(file_knn_pretrained)
file_p_pretrained = open('/dictionaries_submission_pickle/pairwise_pretrained.pickle', 'rb')
p_pretrained = pickle.load(file_p_pretrained)

file_knn_triplets = open('/dictionaries_submission_pickle/knn_triplets.pickle', 'rb')
knn_triplets = pickle.load(file_knn_triplets)
file_p_triplets = open('/dictionaries_submission_pickle/pairwise_triplets.pickle', 'rb')
p_triplets = pickle.load(file_p_triplets)

# x = the one to submit
url = "http://ec2-18-191-24-254.us-east-2.compute.amazonaws.com/results/"

#submit(x, url)
