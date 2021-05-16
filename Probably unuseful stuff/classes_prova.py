import os
import glob
from PIL import Image
import torch.nn.functional as nn
import sklearn
from torchvision import transforms

class Dataset():
    """
    Creates a PyTorch dataset from folder
    Args:
    directory : main directory where images are stored in subfolders.
    """
    def __init__(self, directory):
        self.main_dir = directory
        self.all_subfolders = os.listdir(directory)

    def get_all_images_path(self):
        """
        :return: list of paths for all images
        """
        all_imagespath_list = []
        for image_path in glob.glob(os.path.join(self.main_dir, '*')):
            all_imagespath_list.extend(glob.glob(os.path.join(image_path, '*.jpg')))
        return all_imagespath_list

    def get_all_images(self):
        """
        :return: list of all images as PIL.Image type converted in RGB
        """
        all_img_list = []
        img_path = self.get_all_images_path()
        for img in img_path:
            all_img_list.append((Image.open(img).convert("RGB")))
        return all_img_list


class Encoder():
    """
    Convolutional Encoder
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, (3, 3), padding=(1, 1))
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=(1, 1))
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1))
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d((2, 2))

        self.conv4 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1))
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d((2, 2))

        self.conv5 = nn.Conv2d(128, 256, (3, 3), padding=(1, 1))
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d((2, 2))

    def forward(self, x):
        # Downscale the image with conv, relu and maxpool
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        return x


class Decoder():
    """
    Convolutional Decoder Model
    """

    def __init__(self):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(256, 128, (2, 2), stride=(2, 2))
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(128, 64, (2, 2), stride=(2, 2))
        self.relu2 = nn.ReLU(inplace=True)

        self.deconv3 = nn.ConvTranspose2d(64, 32, (2, 2), stride=(2, 2))
        self.relu3 = nn.ReLU(inplace=True)

        self.deconv4 = nn.ConvTranspose2d(32, 16, (2, 2), stride=(2, 2))
        self.relu4 = nn.ReLU(inplace=True)

        self.deconv5 = nn.ConvTranspose2d(16, 3, (2, 2), stride=(2, 2))
        self.relu5 = nn.ReLU(inplace=True)

    def forward(self, x):
         # Upscale the image with convtranspose and relu
        x = self.deconv1(x)
        x = self.relu1(x)

        x = self.deconv2(x)
        x = self.relu2(x)

        x = self.deconv3(x)
        x = self.relu3(x)

        x = self.deconv4(x)
        x = self.relu4(x)

        x = self.deconv5(x)
        x = self.relu5(x)
        return x


def training_fun(encoder, decoder, data_loader, loss_fn, optimizer):
    """
    Performs a single training step
    Args:
    encoder: A convolutional Encoder. E.g. torch_model ConvEncoder
    decoder: A convolutional Decoder. E.g. torch_model ConvDecoder
    train_loader: PyTorch dataloader, containing (images, images).
    loss_fn: PyTorch loss_fn, computes loss between 2 images.
    optimizer: PyTorch optimizer.
    Returns: Train Loss
    """
    #  Set networks to train mode.
    encoder.train()
    decoder.train()

    for batch_idx, (train_img, target_img) in enumerate(data_loader):

        # Zero grad the optimizer
        optimizer.zero_grad()
        # Feed the train images to encoder
        enc_output = encoder(train_img)
        # Feed the output of encoder to the decoder
        dec_output = decoder(enc_output)

        # Decoder output is reconstructed image
        # Compute loss with it and orginal image (target image before).
        loss = loss_fn(dec_output, target_img)

        # Backpropogate
        loss.backward()

        # Apply the optimizer to network by calling step.
        optimizer.step()
    # Return the loss
    return loss.item()


def create_embedding(encoder, data_loader, embedding_dim):
    """
    Creates embedding using encoder from dataloader.
    encoder: A convolutional Encoder. E.g. torch_model ConvEncoder
    full_loader: PyTorch dataloader, containing (images, images) over entire dataset.
    embedding_dim: Tuple (c, h, w) Dimension of embedding = output of encoder dimesntions.
    Returns: Embedding of size (num_images_in_loader + 1, c, h, w)
    """
    # Set encoder to eval mode.
    encoder.eval()
    # Just a place holder for our 0th image embedding.
    embedding = torch.randn(embedding_dim)

    # No loss so no gradients.
    with torch.no_grad():
        for batch_idx, (train_img, target_img) in enumerate(data_loader):

            # Get encoder outputs and move outputs to cpu
            enc_output = encoder(train_img)
            # Keep adding these outputs to embeddings.
            embedding = torch.cat((embedding, enc_output), 0)

    # Return the embeddings
    return embedding


def compute_similar_images(encoder, image, num_images, embedding):
    """
    Given an image and number of similar images to search.
    Returns the num_images closest nearest images.
    Args:
    image: Image whose similar images are to be found.
    num_images: Number of similar images to find.
    embedding : A (num_images, embedding_dim) Embedding of images learnt from auto-encoder.
    """
    image_tensor = tran.ToTensor()(image)
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        image_embedding = encoder(image_tensor).numpy()

    flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))

    knn = sklearn.NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(embedding)

    _, indices = knn.kneighbors(flattened_embedding)
    indices_list = indices.tolist()
    return indices_list
