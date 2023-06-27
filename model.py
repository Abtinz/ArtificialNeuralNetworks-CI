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
        return isinstance(layer, (Conv2D, MaxPool2D, FC))
        
    def is_activation(self, layer):
        """
        Check if the layer is an activation function.
        args:
            layer: layer to be checked
        returns:
            True if the layer is an activation function, False otherwise
        """
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
        for l in range(len(self.layers_names)):
            layer = self.model[self.layers_names[l]]
            if self.is_layer(layer):
                Z = layer.forward(A)
                tmp.append(Z)
                A = get_activation(self.model[self.layers_names[l + 1]])(Z)
                tmp.append(A)
        
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
        for l in reversed(range(len(self.layers_names))):
            layer = self.model[self.layers_names[l]]
            if self.is_layer(layer):
                if l > 1:
                    Z, A = tmp[l - 1], tmp[l - 2]
                else:
                    Z, A = tmp[l - 1], x
                dZ = dA * get_activation(self.model[self.layers_names[l]]).backward(Z)
                dA, grad = layer.backward(dZ)
                grads[self.layers_names[l]] = grad
        
        return grads

    def update(self, grads):
        """
        Update the model.
        args:
            grads: gradients of the model
        """
        for layer_name, layer in self.model.items():
            if self.is_layer(layer) and not isinstance(layer, MaxPool2D):
                layer.update(grads[layer_name])
    
    def one_epoch(self, x, y):
        """
        One epoch of training.
        args:
            x: input to the model
            y: labels
            batch_size: batch size
        returns:
            loss
        """
        tmp = self.forward(x)
        AL = tmp[-1]
        loss = self.criterion(AL, y)
        dAL = self.criterion.backward()
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
        """
        Load the model.
        args:
            name: name of the model
        returns:
            model, criterion, optimizer, layers_names
        """
        with open(name, 'rb') as f:
            return pickle.load(f)
        
    def shuffle(self, m, shuffling):
        order = list(range(m))
        if shuffling:
            return np.random.shuffle(order)
        return order

    def batch(self, X, y, batch_size, index, order):
        """
        Get a batch of data.
        args:
            X: input to the model
            y: labels
            batch_size: batch size
            index: index of the batch
                e.g: if batch_size = 3 and index = 1 then the batch will be from index [3, 4, 5]
            order: order of the data
        returns:
            bx, by: batch of data
        """
        last_index = min(index + batch_size, X.shape[0])
        batch = order[index:last_index]
        bx = X[batch]
        by = y[batch]
        return bx, by

    def compute_loss(self, X, y, batch_size):
        """
        Compute the loss.
        args:
            X: input to the model
            y: labels
            Batch_Size: batch size
        returns:
            loss
        """
        m = X.shape[0]
        order = np.arange(m)
        cost = 0
        for b in range(m // batch_size):
            bx, by = self.batch(X, y, batch_size, b * batch_size, order)
            AL = self.forward_propagation(bx)
            tmp = self.compute_cost(AL, by)
            cost += tmp
        return cost

    def train(self, X, y, epochs, val=None, batch_size=3, shuffling=False, verbose=1, save_after=None):
        """
        Train the model.
        args:
            X: input to the model
            y: labels
            epochs: number of epochs
            val: validation data
            batch_size: batch size
            shuffling: if True shuffle the data
            verbose: if 1 print the loss after each epoch
            save_after: save the model after training
        """
        train_cost = []
        val_cost = []
        m = X.shape[0]
        for e in tqdm(range(1, epochs + 1)):
            if shuffling:
                order = self.shuffle(X, y)
            else:
                order = np.arange(m)
            cost = 0
            for b in range(m // batch_size):
                bx, by = self.batch(X, y, batch_size, b * batch_size, order)
                AL = self.forward_propagation(bx)
                cost += self.backward_propagation(AL, by)
                self.update_parameters()
            train_cost.append(cost)
            if val is not None:
                val_AL = self.forward_propagation(val[0])
                val_cost.append(self.compute_cost(val_AL, val[1]))
            if verbose != False:
                if e % verbose == 0:
                    print("Epoch {}: train cost = {}".format(e, cost))
                if val is not None:
                    print("Epoch {}: val cost = {}".format(e, val_cost[-1]))
            if save_after is not None and e == save_after:
                self.save("model_after_{}epochs.pickle".format(e))
        return train_cost, val_cost

    def predict(self, X):
        """
        Predict the output of the model.
        args:
            X: input to the model
        returns:
            predictions
        """
        AL = self.forward_propagation(X)
        predictions = np.round(AL)
        return predictions