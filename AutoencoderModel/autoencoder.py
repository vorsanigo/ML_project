import tensorflow as tf
from wandb.keras import WandbCallback


class AutoEncoder():

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
        x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input)
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
        self.autoencoder.compile(optimizer=optimizer, loss=loss, metrics=['mse'])
        #self.autoencoder.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.Accuracy()])
    # Fit  -> not used
    def fit(self, X, n_epochs=50, batch_size=256):
        X_train=X
        self.autoencoder.fit(X_train, X_train, epochs=n_epochs, batch_size=batch_size, shuffle=True)

    def fit2(self, X, n_epochs=50, batch_size=256):
        X_train = X

        # Learning rate scheduler
        def scheduler(n_epochs=50, lr=0.0001):
            # This function keeps the initial learning rate for the first ten epochs
            # and decreases it exponentially after that.
            if n_epochs < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.1)

        callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

        self.autoencoder.fit(X_train,
                             #X_train,
                             epochs=n_epochs,
                             batch_size=batch_size,
                             shuffle=True,
                             callbacks=[callback, WandbCallback()],
                             verbose=1)

    # TODO BEING MODIFIED TO USE DATA AUGMENTATION
    '''def fit2(self, X, n_epochs=50, batch_size=256):
        X_train = X

        # Learning rate scheduler
        def scheduler(n_epochs=50, lr=0.0001):
            # This function keeps the initial learning rate for the first ten epochs
            # and decreases it exponentially after that.
            if n_epochs < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.1)

        callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

        self.autoencoder.fit(X_train, X_train, epochs=n_epochs,
                             batch_size=batch_size,
                             shuffle=True,
                             callbacks=[callback],
                             verbose=1)'''

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




