import tensorflow as tf
from wandb.keras import WandbCallback
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import metrics


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
        self.autoencoder.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'] )

    # Grid search
    def grid_search(self, X_train):

        x_tr, x_ts = train_test_split(X_train, train_size=0.7)
        # Scikit-learn to grid search
        activation = ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'softmax', 'softplus', 'softsign']
        learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
        optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

        # grid search epochs, batch size
        epochs = [1, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        batch_size = [5, 10, 20, 40, 60, 80, 100, 120, 150, 200, 250, 500, 750, 1000, 5000]
        param_grid = dict(epochs=epochs, batch_size=batch_size, activation=activation, optimizer=optimizer)

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
    def fit(self, X, n_epochs=50, batch_size=256):
        X_train = X

        # Learning rate scheduler
        def scheduler(n_epochs=50, lr=0.0001):
            if n_epochs < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.1)

        callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

        self.autoencoder.fit(X_train,
                             epochs=n_epochs,
                             batch_size=batch_size,
                             shuffle=True,
                             callbacks=[callback, WandbCallback()],
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




