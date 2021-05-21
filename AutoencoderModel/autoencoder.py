import tensorflow as tf
from wandb.keras import WandbCallback
from sklearn.model_selection import GridSearchCV, train_test_split


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
        self.autoencoder.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'] )

    # Grid search
    def grid_search(self, X_train):

        x_tr, x_ts = train_test_split(X_train, train_size=0.7)
        # Scikit-learn to grid search
        activation = ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'softmax', 'softplus', 'softsign']
        learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
        optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

        # grid search epochs, batch size
        batch_size = [5, 10, 20, 40, 60, 80, 100, 120, 150, 200, 250, 500, 750, 1000, 5000]
        param_grid = dict(batch_size=batch_size, activation=activation, optimizer=optimizer)

        kmodel = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=self.autoencoder, verbose=1)
        grid = GridSearchCV(estimator=kmodel, param_grid=param_grid, scoring="accuracy", n_jobs=-1, cv=2)
        grid_result = grid.fit(x_tr, x_ts)

        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

    # Fitting
    def fit(self, X, n_epochs=50, batch_size=256, wandb='True'):
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
                                 epochs=n_epochs,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 callbacks=[callback, WandbCallback()],
                                 verbose=1)
        else:
            self.autoencoder.fit(X_train,
                                 epochs=n_epochs,
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




class TripletsEncoder():
    def __init__(self, shape_img, tripletsFile):
        self.shape_img = shape_img
        self.tripletsFile = tripletsFile
        self.triplets_encoder= None

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

        # modellll
        model = tf.keras.models.Model(input, x)
        triplet_model_a = tf.keras.layers.Input(self.shape_img)
        triplet_model_p = tf.keras.layers.Input(self.shape_img)
        triplet_model_n = tf.keras.layers.Input(self.shape_img)
        triplet_model_out = tf.keras.layers.Concatenate()([model(triplet_model_a), model(triplet_model_p), model(triplet_model_n)])
        triplet_model = tf.keras.models.Model([triplet_model_a, triplet_model_p, triplet_model_n], triplet_model_out)

        self.triplets_encoder = triplet_model
        triplet_model.summary()

    def load_triplets(self, triplet_loss, optimizer="adam"):
            print("Loading model...")
            self.triplets_encoder = tf.keras.models.load_model(self.tripletsFile, custum_objects={'loss': triplet_loss(loss)})
            self.triplets_encoder.compile(optimizer=optimizer, loss=triplet_loss)


    def save_triplets(self):
        print("Saving models...")
        self.triplets_encoder.save(self.tripletsFile)

    def compile_triplets(self, triplet_loss, optimizer="adam"):
        self.triplets_encoder.compile(optimizer=optimizer, loss=triplet_loss, metrics=['accuracy'] )


    def fit_triplets(self,data_generator, steps_per_epoch=1, epochs=3):
        self.triplets_encoder.fit(data_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)

    def predict_triplets(self, x):
        return self.triplets_encoder.layers[3].predict(x, verbose=1)






