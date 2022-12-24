def evaluate_model(X, y, n_layers, neurons, lr):
    results = []
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    # define evaluation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
    # define model
    model = build_model(n_inputs, n_outputs,hidden_layer_sizes = n_layers, n_neurons = neurons, lr = lr)
    # enumerate folds
    for train_ix, test_ix in cv.split(X):
        # prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix] 
        # fit model
        model.fit(X_train, y_train, verbose=0, batch_size = 32, epochs=100)
        # evaluate model on test set
        mse_history = model.evaluate(X_test, y_test, verbose=0, batch_size = 32)
        # store result
        #print(mse_history[0])
        results.append(mse_history[0])
    return model, results