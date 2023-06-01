import numpy as np

class MeanSquaredError:
    def __init__(self):
        pass

    # here we are computing the mean square loss error based on real and predicted values and them size!.
    # this formula is came from ./losses/MeanSquaredError.png formula
    #    arguments:
    #        y --> real labels and  batch_size
    #        y_hat --> predicted labels and batch size
    #    returns:  binary cross entropy loss !
    def compute(self, y_pred, y_true):
        
        #(y_pred.shape[1] -> batch_size)
        cost = (1 / (2 * y_pred.shape[1])) * np.sum((y_pred - y_true) ** 2)
        #we are using np.squeeze on our function because error dimension is 1x1 ...
        return np.squeeze(cost)
    
    def backward(self, y_pred, y_true):

        return (1 / y_pred.shape[1]) * (y_pred - y_true) #(y_pred.shape[1] -> batch_size)