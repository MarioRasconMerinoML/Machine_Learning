def plot_validation_results(results, model_dir, model):
    '''
    It plots the MSE for the different folds used in validation
    
    results --> a list of the metric employed along different dataset splits via KFold
    model_dir --> path to store the plot
    model --> we will use model.name in order to identify the plots
    '''
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    plt.plot(pd.DataFrame(results))
    plt.grid(True)
    plt.gca().set_ylim(0, 0.3)
    plt.gca().set_xlim(-0.2, 50)
    plt.axhline(np.mean(results), color = 'blue',linestyle = '--')
    plt.xlabel('Fold')
    plt.ylabel ('MSE')
    title = 'Validation MSE per fold for ' + model.name
    plt.title(title)
    path = os.path.join(model_dir, title + ".png" )
    plt.savefig(path, dpi=500)
    plt.show()
    plt.close()