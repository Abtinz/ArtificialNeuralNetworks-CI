import numpy as np

class BinaryCrossEntropy:
    def __init__(self) -> None:
        pass

    # here we are computing the binary cross entropy loss error based on real and predicted values and them size!.
    # this formula is came from ./losses/binaryCrossEntropy.png formula
    #    arguments:
    #        y --> real labels and  batch_size
    #        y_hat --> predicted labels and batch size
    #    returns:  binary cross entropy loss !
    def compute(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        
        epsilon = 1e-7  # small value to avoid division by zero, its not to small for process!

        #cost function formula in binary cross entropy -> ./losses/binaryCrossEntropy.png formula
        cost = ((-1/  y.shape[1]) * np.sum(y * np.log(y_hat + epsilon) + (1 - y) * np.log(1 - y_hat + epsilon)))
        
        return np.squeeze(cost)

    
    # this function will compute the derivative of the binary cross entropy loss.
    #  arguments:
    #    y --> real labels and  batch_size
    #    y_hat --> predicted labels and batch size
    #  returns:  binary cross entropy loss !
    def backward(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
       
        epsilon =1e-7 
        dZ = (y_hat - y) / (y_hat * (1 - y_hat) + epsilon)
        return dZ

