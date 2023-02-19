def plot_train_val_loss_acc(history):
    # Plot model train and validation curves per epoch and saves the plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2,1, figsize = (12,10))
    
    accuracy = history.history['accuracy']
    val_accuracy =  history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(accuracy) + 1)
    ax[0].plot(epochs, accuracy, "bo", label = "Training Accuracy")
    ax[0].plot(epochs, val_accuracy, "rx", label = "Validation Accuracy")
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].grid(True)
    ax[0].legend(loc='upper right')
    title1 = 'Train - Validation Accuracy curves for ' + history.model.name
    
    ax[0].set_title(title1)
    
    ax[1].plot(epochs, loss, "bo", label = "Training Loss")
    ax[1].plot(epochs, val_loss, "rx", label = "Validation Loss")
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')

    ax[1].grid(True)
    ax[1].legend(loc='upper right')
    title2 = 'Train - Validation Loss curves for ' + history.model.name
    
    ax[1].set_title(title2)
    
    #path = os.path.join(model_dir, title + ".png" )
    #plt.savefig(path, dpi=500)
    plt.show()