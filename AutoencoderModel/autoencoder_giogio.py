import tensorflow as tf
from wandb.keras import WandbCallback
from sklearn.model_selection import GridSearchCV, train_test_split


class AutoEncoder:

    def __init__(self, shape_img, autoencoderFile, encoderFile):
        self.shape_img = shape_img
        self.autoencoderFile = autoencoderFile
        self.encoderFile = encoderFile
        self.autoencoder = None
        self.encoder = None

    # Inference
    def predict(self, X):
        return self.encoder.predict(X)

    # Set neural network architecture
    def set_arch(self):

        # Set encoder and decoder graphs
        input = tf.keras.layers.Input(shape=self.shape_img)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

        x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        decoded = tf.keras.layers.Conv2D(self.shape_img[2], (3, 3), activation='softmax', padding='same')(x)

        # Create and save models
        autoencoder = tf.keras.Model(input, decoded)
        encoder = tf.keras.Model(input, encoded)

        self.autoencoder = autoencoder
        self.encoder = encoder

        # Generate summaries
        print("\nautoencoder.summary():")
        print(autoencoder.summary())
        print("\nencoder.summary():")
        print(encoder.summary())

    # Compile
    def compile(self, loss="mse", optimizer="adam"):
        self.autoencoder.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Fitting
    def fit(self, X, steps_per_epoch=120, n_epochs=500, batch_size=64, wandb='True'):
        X_train = X

        # Learning rate scheduler
        def scheduler(n_epochs, lr=0.0001):
            if n_epochs < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.1)

        callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

        if wandb == 'True':
            self.autoencoder.fit(X_train,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=n_epochs,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 callbacks=[callback, WandbCallback()],
                                 verbose=1)
        else:
            self.autoencoder.fit(X_train,
                                 epochs=n_epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 callbacks=[callback],
                                 verbose=1)

    # Save model architecture and weights to file
    def save_models(self):
        print("Saving models...")
        self.autoencoder.save(self.autoencoderFile)
        self.encoder.save(self.encoderFile)

    # Load model architecture and weights
    def load_models(self, loss="mse", optimizer="adam"):
        print("Loading model...")
        self.autoencoder = tf.keras.models.load_model(self.autoencoderFile)
        self.encoder = tf.keras.models.load_model(self.encoderFile)
        self.autoencoder.compile(optimizer=optimizer, loss=loss)
        self.encoder.compile(optimizer=optimizer, loss=loss)


class TripletsEncoder:
    def __init__(self, shape_img, tripletsFile):
        self.shape_img = shape_img
        self.tripletsFile = tripletsFile
        self.triplets_encoder = None

    # Inference
    def predict_triplets(self, x):
        return self.triplets_encoder.layers[3].predict(x, verbose=1)

    # Set neural network architecture
    def set_arch(self):
        input = tf.keras.layers.Input(shape=self.shape_img)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(100, activation='relu')(x)

        # Model
        model = tf.keras.models.Model(input, x)
        triplet_model_a = tf.keras.layers.Input(self.shape_img)
        triplet_model_p = tf.keras.layers.Input(self.shape_img)
        triplet_model_n = tf.keras.layers.Input(self.shape_img)
        triplet_model_out = tf.keras.layers.Concatenate()([model(triplet_model_a), model(triplet_model_p), model(triplet_model_n)])
        triplet_model = tf.keras.models.Model([triplet_model_a, triplet_model_p, triplet_model_n], triplet_model_out)

        self.triplets_encoder = triplet_model
        triplet_model.summary()

    # Compiling
    def compile_triplets(self, triplet_loss, optimizer="adam"):
        self.triplets_encoder.compile(optimizer=optimizer, loss=triplet_loss, metrics=['accuracy'])

    # Fitting
    def fit_triplets(self, data_generator, steps_per_epoch=1, epochs=3, batch_size=256, wandb='True'):

        # Learning rate scheduler
        def scheduler(epochs, lr=0.0001):
            if epochs < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.1)

        callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

        if wandb == 'True':
            self.triplets_encoder.fit(data_generator,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      callbacks=[callback, WandbCallback()],
                                      verbose=1)
        else:
            self.triplets_encoder.fit(data_generator,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      callbacks=[callback],
                                      shuffle=True,
                                      verbose=1)

        #original:
        #self.triplets_encoder.fit(data_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)

    # Save model architecture and weights to file
    def save_triplets(self):
        print("Saving models...")
        self.triplets_encoder.save(self.tripletsFile)

    # Load model architecture and weights
    def load_triplets(self, triplet_loss, optimizer="adam"):
        print("Loading model...")
        self.triplets_encoder = tf.keras.models.load_model(self.tripletsFile, custom_objects={'triplet_loss':triplet_loss})
        self.triplets_encoder.compile(optimizer=optimizer, loss=triplet_loss)
