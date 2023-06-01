import numpy as np

class MeanSquaredError:
    def __init__(self):
        pass

    def compute(self, y_pred, y_true):
        """
        computes the mean squared error loss
            args:
                y_pred: predicted labels (n_classes, batch_size)
                y_true: true labels (n_classes, batch_size)
            returns:
                mean squared error loss
        """
        # Implement mean squared error loss
        batch_size = y_pred.shape[1]
        cost = (1 / (2 * batch_size)) * np.sum((y_pred - y_true) ** 2)
        return np.squeeze(cost)
    
    def backward(self, y_pred, y_true):
        """
        computes the derivative of the mean squared error loss
            args:
                y_pred: predicted labels (n_classes, batch_size)
                y_true: true labels (n_classes, batch_size)
            returns:
                derivative of the mean squared error loss
        """
        # Implement backward pass for mean squared error loss
        batch_size = y_pred.shape[1]
        d_loss = (1 / batch_size) * (y_pred - y_true)
        return d_loss