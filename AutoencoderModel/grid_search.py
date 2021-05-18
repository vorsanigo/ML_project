from sklearn.model_selection import GridSearchCV

def grid_search(model, X):
    # Use scikit-learn to grid search
    activation = ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'softmax', 'softplus', 'softsign']
    learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

    # grid search epochs, batch size
    epochs = [1, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    batch_size = [5, 10, 20, 40, 60, 80, 100, 120, 150, 200, 250, 500, 750, 1000, 5000]
    param_grid = dict(epochs=epochs, batch_size=batch_size)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(X)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
