import numpy as np
from abc import ABC, abstractmethod

class Activation:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        Forward pass for activation function.
            args:
                Z: input to the activation function
            returns:
                A: output of the activation function
        """
        pass

    @abstractmethod
    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Backward pass for activation function.
            args:
                dA: derivative of the cost with respect to the activation
                Z: input to the activation function
            returns:
                derivative of the cost with respect to Z
        """
        pass

class Sigmoid(Activation):
    def forward(self, Z: np.ndarray) -> np.ndarray:
        return (1 / (1 + np.exp(-Z)))

    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        
        A = self.forward(Z)
        dZ = dA *(A * (1 - A)) 
        return dZ
    
 #relu forward -> z >= 0 : Z and Z < 0: 0
class ReLU(Activation):
    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        ReLU activation function.
            args:
                x: input to the activation function
            returns:
                relu(x)
        """
       
        A = np.maximum(0, Z)
        return A

    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Backward pass for ReLU activation function.
            args:
                dA: derivative of the cost with respect to the activation
                Z: input to the activation function
            returns:
                derivative of the cost with respect to Z
        """
        
        dZ = np.copy(dA)
        dZ[Z <= 0] = 0 
        return dZ
    

class Tanh(Activation):
    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        Tanh activation function.
            args:
                x: input to the activation function
            returns:
                tanh(x)
        """
        return np.tanh(Z)

    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Backward pass for tanh activation function.
            args:
                dA: derivative of the cost with respect to the activation
                Z: input to the activation function
            returns:
                derivative of the cost with respect to Z
        """
        #d(tanh) -> 1 - square of forward resualt  
        return dA * (1 - np.square(self.forward(Z)))
    
class LinearActivation(Activation):
    """
        Linear activation function. the most simple activation function in the world!
        A  = Z    
    """
    def linear(Z: np.ndarray) -> np.ndarray:
        return Z

    def backward(dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        #(dx/dx)*dA = 1 * dA = dA
        return dA

def get_activation(activation: str) -> tuple:
    """
    Returns the activation function and its derivative.
        args:
            activation: activation function name
        returns:
            activation function and its derivative
    """
    if activation == 'sigmoid':
        return Sigmoid
    elif activation == 'relu':
        return ReLU
    elif activation == 'tanh':
        return Tanh
    elif activation == 'linear':
        return LinearActivation
    else:
        raise ValueError('Activation function not supported')