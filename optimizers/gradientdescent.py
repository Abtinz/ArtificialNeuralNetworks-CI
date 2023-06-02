#Gradient Descent optimizer.
#this class will help us to implement gradient descent 
#slf parameters -> learning rate and layers_list : dictionary of layers name and layer object

class GD:

    def __init__(self, layers_list: dict, learning_rate: float):
        self.learning_rate = learning_rate
        self.layers = layers_list
    
    
    #Update the parameters of the layer.
    #formula -> ./optimizers/GradientDescent.png
    #  arguments:
    #      grads: list of gradients for the weights and bias
    #      name: name of the layer we want to update with gradient descent
    #  returns: params: list of updated parameters
    
    def update(self, grads, name):
       
        layer = self.layers[name]
        updated_parameters = []
        # Update the weights and bias using gradient descent
        for index in range(len(grads)):
            updated_parameters.append((layer.params[index] - self.learning_rate * grads[index]))
        return updated_parameters