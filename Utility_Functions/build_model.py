def build_model(n_inputs, n_outputs, hidden_layer_sizes = 2, n_neurons = 128, lr = 1e-3):
    
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape = n_inputs))
    for layer in range(hidden_layer_sizes):
        model.add(keras.layers.Dense(n_neurons, activation = 'relu'))
    # Output Layer
    model.add(Dense(n_outputs))
    optim = keras.optimizers.Adam(learning_rate = lr)
    model.compile(loss='mse', 
                  optimizer= optim, 
                  metrics =[tf.keras.metrics.MeanSquaredError()])
    return model