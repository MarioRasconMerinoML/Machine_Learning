


def plot_decision_regions(X, y, classifier, resolution = 0.02, test_idx = None, dataset_name = None):
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    custom_cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # Plot decision region
    # Coordinates of the straight line from results
    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
    
    # Generate a mesh around those points
    xx1,xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    
    #Label prediction
    y_pred = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab_pred = y_pred.reshape(xx1.shape)
    
    
    
    # plot samples
    plt.figure(figsize=(10, 5))
    for idx, cls in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cls, 0], y = X[y == cls, 1], alpha = 0.8, c = colors[idx], marker = markers[idx], label = f'Class {cls}', edgecolor = 'black')
    
    # Plot surface decision boundaries
    plt.contourf(xx1, xx2, lab_pred, alpha = 0.3, cmap = custom_cmap)
    
    #Label probability
    # Note SVM classifiers do not output probabilities like Logistic ones
    if hasattr(classifier, 'predict_proba'):
        y_prob = classifier.predict_proba(np.c_[xx1.ravel(), xx2.ravel()])
        lab_prob = y_prob[:,1].reshape(xx1.shape)
        contour = plt.contour(xx1, xx2, lab_prob, alpha = 0.9, cmap = plt.cm.brg)
        plt.clabel(contour, inline = 1, fontsize = 9)
        
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
        
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:,0], X_test[:,1],
                          c = 'none', edgecolor = 'black',
                          alpha = 1.0, lw = 1, marker = 'o', s = 100,  label = 'Test Set')
                          
    
    plt.legend(loc = 'center left')
    plt.title(str(classifier) + ' ' +'contour_plot')
    plt.grid(True)
    plt.tight_layout()
    
    # Comment / change for future use
    '''
    # Iris Dataset
    if dataset_name == 'Iris':
        plt.xlabel("Petal length [standarized]", fontsize=14)
        plt.ylabel("Petal width [standarized]", fontsize=14)
    elif dataset_name == 'Wine_PCA':
        plt.xlabel("PC1", fontsize=14)
        plt.ylabel("PC2", fontsize=14)
    else:
        pass
    '''
    plt.show()