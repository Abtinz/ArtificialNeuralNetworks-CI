from layers.convolution2d import Conv2D
from layers.maxpooling2d import MaxPool2D
from layers.fullyconnected import FC

from activations import Activation, get_activation

import pickle
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

class Model:
    def __init__(self, arch, criterion, optimizer, name=None):
        """
        Initialize the model.
        args:
            arch: dictionary containing the architecture of the model
            criterion: loss 
            optimizer: optimizer
            name: name of the model
        """
        if name is None:
            self.model = arch
            self.criterion = criterion
            self.optimizer = optimizer
            self.layers_names = list(arch.keys())
        else:
            self.model, self.criterion, self.optimizer, self.layers_names = self.load_model(name)
    
    def is_layer(self, layer):
        """
        Check if the layer is a layer.
        args:
            layer: layer to be checked
        returns:
            True if the layer is a layer, False otherwise
        """
        # Check if the layer is an instance of the Conv2D, MaxPool2D, or FC class
        return isinstance(layer, (Conv2D, MaxPool2D, FC))

    def is_activation(self, layer):
        """
        Check if the layer is an activation function.
        args:
            layer: layer to be checked
        returns:
            True if the layer is an activation function, False otherwise
        """
        # Check if the layer is an instance of the Activation class or its subclasses
        return isinstance(layer, Activation)

    def forward(self, x):
        """
        Forward pass through the model.
        args:
            x: input to the model
        returns:
            output of the model
        """
        tmp = []
        A = x
        # Forward pass through the model
        # We have a pattern of layers and activations
        for l in range(len(self.layers_names)):
            layer = self.model[self.layers_names[l]]
            if self.is_layer(layer):
                Z = layer.forward(A)
                tmp.append(Z.copy())
                A = Z
            elif self.is_activation(layer):
                A = layer.forward(Z)
                tmp.append(A.copy())
        return tmp
    
    def backward(self, dAL, tmp, x):
        """
        Backward pass through the model.
        args:
            dAL: derivative of the cost with respect to the output of the model
            tmp: list containing the intermediate values of Z and A
            x: input to the model
        returns:
            gradients of the model
        """
        dA = dAL
        grads = {}
        # Backward pass through the model
        # We have a pattern of layers and activations
        # We go from the end to the beginning of the tmp list
        for l in range(len(self.layers_names), 0, -1):
            layer = self.model[self.layers_names[l - 1]]
            if self.is_layer(layer):
                Z, A = tmp[l - 1], tmp[l - 2]
                dZ = dA * layer.backward(Z)
                dA, grad = layer.backward(dZ, A)
                grads[self.layers_names[l - 1]] = grad
            elif self.is_activation(layer):
                A = tmp[l - 2]
                dA = layer.backward(dA, Z)
        return grads

    def update(self, grads):
        """
        Update the model.
        args:
            grads: gradients of the model
        """
        for layer_name in self.layers_names:
            layer = self.model[layer_name]
            if self.is_layer(layer) and not isinstance(layer, MaxPool2D):
                layer.update(grads[layer_name], self.optimizer)
    
    def one_epoch(self, x, y):
        """
        One epoch of training.
        args:
            x: input to the model
            y: labels
        returns:
            loss
        """
        # One epoch of training
        tmp = self.forward(x)
        AL = tmp[-1]
        loss = self.criterion(AL, y)
        dAL = self.criterion.backward(AL, y)
        grads = self.backward(dAL, tmp, x)
        self.update(grads)
        return loss
    
    def save(self, name):
        """
        Save the model.
        args:
            name: name of the model
        """
        with open(name, 'wb') as f:
            pickle.dump((self.model, self.criterion, self.optimizer, self.layers_names), f)
    
    def load_model(self, name):
    # """
    # Load the model.
    # args:
    #     name: name of the model
    # returns:
    #     model, criterion, optimizer, layers_names
    # """
        with open(name, 'rb') as f:
            model, criterion, optimizer, layers_names = pickle.load(f)
        return model, criterion, optimizer, layers_names
                    