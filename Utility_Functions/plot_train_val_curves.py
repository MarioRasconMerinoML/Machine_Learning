def plot_train_val_curves(history, model_dir):
    # Plot model train and validation curves per epoch and saves the plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.gca().set_xlim(0,50)
    plt.gca().set_ylim(0, 1)
    plt.grid(True)
    plt.legend(['train', 'val'], loc='upper right')
    title = 'Train - Validation MSE curves for ' + history.model.name
    plt.title(title)
    path = os.path.join(model_dir, title + ".png" )
    plt.savefig(path, dpi=500)
    plt.show()
    plt.close()
    