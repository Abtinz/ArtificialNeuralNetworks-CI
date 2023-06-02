import numpy as np
from .gradientdescent import GradientDescent

class Adam:
    def __init__(self, layers_list, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = layers_list
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.V = {}
        self.S = {}
        for name in layers_list:
            # Initialize V and S for each layer (v and s are lists of zeros with the same shape as the parameters)
            v = [np.zeros_like(parameter) for parameter in layers_list[name].parameters]
            s = [np.zeros_like(parameter) for parameter in layers_list[name].parameters]
            self.V[name] = v
            self.S[name] = s
        
    def update(self, grads, name, epoch):
        layer = self.layers[name]
        parameters = []
        # Adam update -> param = param - learning_rate * V_corrected / (sqrt(S_corrected) + epsilon).
        for i in range(len(grads)):
            #V correction
            self.V[name][i] = self.beta1 * self.V[name][i] + (1 - self.beta1) * grads[i]
            V_corrected = self.V[name][i] / (1 - np.power(self.beta1, epoch))
            #S correction
            self.S[name][i] = self.beta2 * self.S[name][i] + (1 - self.beta2) * np.square(grads[i])           
            S_corrected = self.S[name][i] / (1 - np.power(self.beta2, epoch))

            parameter = layer.parameters[i] - self.learning_rate * V_corrected / (np.sqrt(S_corrected) + self.epsilon)
            parameters.append(parameter)
        return parameters