from math import sqrt
import numpy as np

class FC:

    # name -> a string identifier for the layer, we will use it for distinguishing con layers from fully connected layers
    # self initialize_method is used in weights initializing ...
    def __init__(self, input_size : int, output_size : int, name : str, initialize_method : str="random"):
        self.input_size = input_size
        self.output_size = output_size
        self.name = name
        self.initialize_method = initialize_method
        self.parameters = [self.initialize_weights(), self.initialize_bias()]
        self.input_shape = None
        self.reshaped_shape = None
    
    def initialize_weights(self):

        if self.initialize_method == "random":  return np.random.randn(self.input_size, self.output_size)

        #This method initializes the weights with random numbers drawn from a uniform distribution with a specific range.
        elif self.initialize_method == "xavier":
            #uniform range
            range_value = np.sqrt( 1 / (self.input_size + self.output_size))
            return np.random.uniform(-range_value, range_value, (self.input_size, self.output_size))
         

        elif self.initialize_method == "he":
            #using gaussian distribution
            return np.random.randn((self.input_size, self.output_size)) *  sqrt(2 / self.input_size)

        else:
            raise ValueError("Invalid initialization method")

    #Initialize bias with zeros and layer out put count which will show the number of our perceptrons      
    def initialize_bias(self):
        return np.zeros((self.output_size, 1))
    
    #here, forwarding in our network is implemented for this layer
    #A_prev -> the are activations that reached from previous layer(they are layer inputs)
    #A_prev.shape returns (batch_size, input_size) so we can know the output size of our last layer which is input size of current layer
    #it will return a out put of this fully connected forwarding
    def forward(self, A_prev):
        
        self.input_shape = A_prev.shape
        A_prev_tmp = np.copy(A_prev)

        if len(A_prev_tmp.shape) > 2: # check if A_prev is output of convolutional layer
            batch_size = A_prev_tmp.shape[0]
            A_prev_tmp = A_prev_tmp.reshape(batch_size, -1).T
            self.reshaped_shape = A_prev_tmp.shape
    
        #Forward part
        weight, bias = self.parameters
        # weight.T = weight_transpose
        # output X(input) * weight_transpose +bias
        output = np.dot(weight.T, A_prev_tmp) + bias
        return output
  
    # Backward pass for fully connected layer.
    #    args:
    #         dZ: derivative of the cost with respect to the output of the current layer
    #         A_prev: activations from previous layer (or input data)
    #     returns:
    #         dA_prev: derivative of the cost with respect to the activation of the previous layer
    #         grads: list of gradients for the weights and bias
    def backward(self, dZ, A_prev):
   
        A_prev_tmp = np.copy(A_prev)
        if len(A_prev_tmp.shape) > 2: # check if A_prev is output of convolutional layer
            batch_size = A_prev_tmp.shape[0]
            A_prev_tmp = A_prev_tmp.reshape(batch_size, -1).T

        # backward part
        weight, bias = self.parameters
        dW = np.dot(A_prev_tmp, dZ.T) / A_prev_tmp.shape[1]
        db = np.sum(dZ, axis=1, keepdims=True) / A_prev_tmp.shape[1]
        dA_prev = np.dot(weight, dZ)
        grads = [dW, db]
        
        # reshape dA_prev to the shape of A_prev
        if len(A_prev.shape) > 2:    # check if A_prev is output of convolutional layer
            dA_prev = dA_prev.T.reshape(self.input_shape)
        return dA_prev, grads
    
    def update_parameters(self, optimizer, grads):
        self.parameters = optimizer.update(grads, self.name)