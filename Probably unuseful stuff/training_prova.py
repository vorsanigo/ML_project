# Simplified Training Script
from classes_prova import *
import torch
import torch.nn.functional as nn
import tqdm
import numpy as np

#transforms = tran.Compose([tran.ToTensor()])  # Normalize the pixels and convert to tensor.

path = "C:/Users/utente.000/Desktop/APPLIED MACHINE LEARNING/Project/prova"
df = Dataset(path)  # Create folder dataset

# Create dataloader
data_loader = torch.utils.data.DataLoader(df, batch_size=32, shuffle=True)

loss_fn = nn.MSELoss()  #MSE to compute distance btw images

encoder = Encoder()  # The encoder
decoder = Decoder()  # The decoder

# Both the encoder and decoder parameters
autoencoder_params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(autoencoder_params, lr=1e-3)
# Adam Optimizer (alternative to gradient descent to update weights)

# Actual training
EPOCHS = 10
max_loss = 0

for epoch in tqdm(range(EPOCHS)):
    train_loss = training_fun(encoder, decoder, data_loader, loss_fn, optimizer)
    print(f"Epochs = {epoch}, Training Loss : {train_loss}")


# Save the feature representations.
EMBEDDING_SHAPE = (1, 256, 16, 16)  # Known from the encoder

embedding = create_embedding(encoder, data_loader, EMBEDDING_SHAPE)

# Convert embedding to numpy and save them
numpy_embedding = embedding.numpy()
num_images = numpy_embedding.shape[0]

# Save the embeddings for complete dataset, not just train
flattened_embedding = numpy_embedding.reshape((num_images, -1))
np.save("data_embedding.npy", flattened_embedding)

print("Done")
