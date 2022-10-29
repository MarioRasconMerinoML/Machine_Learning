class SimpleMLP(kt.HyperModel):
    ## ML Libraries
    # Class to ease Hypertunning with Keras Tuner
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        #self.model_name = model_name

    def build(self, hp):
        model = keras.Sequential(name = 'Optimized')
        model.add(keras.layers.InputLayer(input_shape = self.num_inputs))
        units = hp.Int(name="units", min_value=4, max_value=64, step=4)
        for layer in range(hp.Int('num_layers',0,4)):
            model.add(keras.layers.Dense(units = units, activation = 'relu'))
            # Tune whether to use dropout.

        model.add(layers.Dropout(hp.Float('dropout', 0, 0.3, step=0.1, default=0.2)))    
        model.add(layers.Dense(self.num_outputs))    
        optimizer = hp.Choice(name="optimizer", values=["adam"])
        model.compile(optimizer=optimizer, loss="mse",metrics=[tf.keras.metrics.MeanSquaredError()])
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")    
        
        return model