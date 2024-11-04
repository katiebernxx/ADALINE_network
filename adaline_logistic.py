'''
Katie Bernard
Logistic regression version of ADALINE
'''
import numpy as np
from adaline import Adaline
        
class AdalineLogistic(Adaline):
    def __init__(self):
        super().__init__()
    def activation(self, net_in):
        '''Applies the activation function to the net input and returns the output neuron's activation.
        sigmoid activation

        Parameters:
        ----------
        net_in: ndarray. Shape = [Num samples N,]

        Returns:
        ----------
        net_act. ndarray. Shape = [Num samples N,]
        '''
        net_act = 1 / (1 + np.exp(-net_in))

        return net_act
        
    def predict(self, features):
        '''Predicts the class of each test input sample

        Parameters:
        ----------
        features: ndarray. Shape = [Num samples N, Num features M]
            Collection of input vectors.

        Returns:
        ----------
        The predicted classes (1 or 0) for each input feature vector. Shape = [Num samples N,]

        '''
        # net input --> weighted sum
        net_in = self.net_input(features)

        # net activation 
        net_act = self.activation(net_in)

        #predict
        predicted_classes = np.where(net_act >=0.5, 1, 0)

        #return predicted classes
        return predicted_classes
        
    def loss(self, y, net_act):
        ''' Computes the cross entropy loss (over a single training epoch)

        Parameters:
        ----------
        y: ndarray. Shape = [Num samples N,]
            True classes corresponding to each input sample in a training epoch (coded as -1 or +1).
        net_act: ndarray. Shape = [Num samples N,]
            Output neuron's activation value (after activation function is applied)

        Returns:
        ----------
        float. The SSE loss (across a single training epoch).
        '''
        #To prevent log(0)
        epsilon = 1e-15
        net_act = np.clip(net_act, epsilon, 1 - epsilon)
        
        loss = -np.sum(y * np.log(net_act) + (1 - y) * np.log(1 - net_act))
        return loss
