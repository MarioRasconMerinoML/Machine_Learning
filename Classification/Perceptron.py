# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 11:44:50 2022

@author: C40656
"""

import numpy as np

'''
Implementation of a Perceptron Classifier in pure Python from S.Raschka Book

'''
class Perceptron:
    
    '''
    Perceptron Linear Classifier
    
    Parameters:
        
        eta : learning rate (float) (normally between 0 and 1)
        n_iter : iterations over training set (int)
        random_state: random generatos seed for random weight initialization (int)
    
    Attributes:
        (Attributes not created upon initialization like the ones with self.)
        
        w_ : 1d array (weights after fitting)
        b_ : scalar (Bias Unit after fitting)
        errors_ : number of misclassifications after each epoch update
    
    '''
    
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 42):
        
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        
        ''' Fit Training Data 
        
        Parameters: 
            
            X: (array), shape = [n_examples, n_features], training vectors
            y: (array), shape = [n_examples], targets
            
        Returns:
            
            self: object
        '''
        
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = X.shape[1])
        self.b_ = np.float_(0.0)
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            
            self.errors_.append(errors)
        return self
    
    def predict (self, X):
        ''' return class label '''
        
        self.pred_ = np.dot(X, self.w_) + self.b_
        
        return np.where(self.pred_ >= 0.0, 1, 0)
    
                
        
        
        