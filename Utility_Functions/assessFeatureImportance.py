def assessFeatureImportance(df_features,df_targets, score_func = None):
    '''
    Inputs are 2 dataframes with the splitted features and targets
    
    Example:
    
    from sklearn.feature_selection import mutual_info_regression, r_regression
    assessFeatureImportance(features,targets,mutual_info_regression)
    '''
    from sklearn.feature_selection import mutual_info_regression, r_regression, SelectKBest
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import numpy as np
    #%matplotlib inline
    
    def select_features(X_train_f, y_train_f, X_test_f):
        # configure to select all features
        fs = SelectKBest(score_func = score_func, k='all')
        # learn relationship from training data
        fs.fit(X_train_f, y_train_f)
        # transform train input data
        X_train_fs = fs.transform(X_train_f)
        # transform test input data
        X_test_fs = fs.transform(X_test_f)
        return X_train_fs, X_test_fs, fs
    
    fig, axs = plt.subplots(ncols=len(df_targets.columns), figsize=(len(df_targets.columns)*6, 6),
                        constrained_layout=True)
    axs = np.ravel(axs)
    # EXtract numerical Values
    X = df_features.values
    y = df_targets.values
    
    for ind, col in enumerate(df_targets.columns):
        # Select target
        X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X, y[:,ind], test_size = 0.3, random_state = 0)
        # feature selection
        X_train_fs, X_test_fs, fs = select_features(X_train_f, y_train_f, X_test_f)
        print(fs.scores_)
        # plot the scores
        axs[ind].bar([feature for feature in df_features.columns], fs.scores_)
        axs[ind].set_title('Feature importance in ' + df_targets.columns[ind])
    plt.show()