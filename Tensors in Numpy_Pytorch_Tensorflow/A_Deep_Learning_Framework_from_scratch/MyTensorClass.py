# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 21:29:55 2022

@author: MRSC

From Grokkking Deep Learning. A. Trask

"""

import numpy as np

class MyTensorClass(object):
    '''
    x = MyTensorClass([2,3,5,1,7], autograd = True)
    y = MyTensorClass([3,1,9,2,4], autograd = True)
    
    z = x + y
    z.backward(MyTensorClass(np.array([1,1,1,1,1])))
    
    print(y.grad.data == np.array([1,1,1,1,1]))
    '''
    def __init__(self, data,autograd = False,
                 new_edge = None, new_node = None,
                 id = None):
        self.data = np.array(data) # it receives a numpy arry to create the tensor
        '''
        We implement the creation of a computational graph
        '''
        self.new_edge = new_edge # graph creation op
        self.new_node = new_node # node creation
        self.grad = None
        
        # Implement the expansion for the graph to perform automatic gradient
        # This is similar to Pytorch implementation
        
        self.autograd = autograd
        self.children = {} # Dictionary to store weights of nodes
        
        # Generate nodes unique identifier
        if (id is None):
            id = np.random.randint(0,100000)
        self.id = id
        
        if (new_node is not None):
            for node in new_node:
                if (self.id not in node.children): # keeps track how many children a tensor has
                    node.children[self.id] = 1
                else:
                    node.children[self.id] += 1
    
    def counter_of_grads_in_children_nodes(self):
        # Counter for the correct number of gradients received from each child
        # during 
        for id,cnt in self.children.items():
            if (cnt != 0):
                return False
            return True
    
    # Add some methods, typically from matrices operations, add, substract...
    def __add__(self, other):
        # Method to add nodes to the graph
        if(self.new_edge == 'neg'):
            self.new_node[0].backward(self.grad.__neg__())
        if (self.autograd and other.autograd):
            return MyTensorClass(self.data + other.data,
                                 autograd = True,
                                 new_node = [self,other],
                                 new_edge = "add") # Type of new graph operation, add in this case
        return MyTensorClass(self.data + other.data)
    
    ## Substraction
    def __substract__(self, other):
        return MyTensorClass(self.data - other.data)
    
    def __repr__(self):
        return str(self.data.__repr__()) # To print the tensor
    
    def __str__(self):
        return str(self.data.__str__())
    
    def backward(self, grad = None, grad_origin = None):
        # Graph creation for backprop
        # self.autograd to True performs gradient calculation and then 
        # training in the backprop stage not during forward pass
        # This is how Pytorch works for instance
        
        if (self.autograd):
            if (grad_origin is not None):
                if (self.children[grad_origin.id] == 0):
                    raise Exception("Cannot backprop more than once")
                else:
                    self.children[grad_origin.id] -= 1
            
            if (self.grad is None):
                self.grad = grad
            else:
                self.grad += grad
            
            if(self.new_node is not None and (self.counter_of_grads_in_children_nodes() or 
                                              grad_origin is None)):
         
                if (self.new_edge == 'add'):
                    # create new nodes (2 in this case) Note --> Extend to n
                    self.new_node[0].backward(self.grad, self) # Gets backprop for the new node
                    self.new_node[1].backward(self.grad, self)
    def __neg__(self):
        # Negation suport
        
        if (self.autograd):
            return MyTensorClass(self.data * -1,
                                 autograd = True,
                                 new_node = [self],
                                 new_edge = "neg")
        return MyTensorClass(self.data * -1)
    
    
    
    ## Multiplication
    
    ## sum
    
    ## expand 
    
    ## Transpose
    
    # Dot Product