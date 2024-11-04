'''
Katie Bernard
ADALINE (ADaptive LInear NEuron) neural network for classification and regression
'''
import numpy as np

class Adaline():
    ''' Single-layer neural network

    Network weights are organized [wt1, wt2, wt3, ..., wtM] for a net with M input neurons.
    Bias is stored separately from wts.
    '''
    def __init__(self):
        '''ADALINE Constructor
        '''
        # Network weights: wt for input neuron 1 is at self.wts[0], wt for input neuron 2 is at self.wts[1], etc
        self.wts = None
        # Bias: will be a scalar
        self.b = None
        # Record of training loss. Will be a list. Value at index i corresponds to loss on epoch i.
        self.loss_history = None
        # Record of training accuracy. Will be a list. Value at index i corresponds to acc. on epoch i.
        self.accuracy_history = None

    def get_wts(self):
        ''' Returns a copy of the network weight array'''
        return self.wts

    def get_bias(self):
        ''' Returns a copy of the bias'''
        return self.b

    def net_input(self, features):
        ''' Computes the net_input (weighted sum of input features,  wts, bias)

        Parameters:
        ----------
        features: ndarray. Shape = [Num samples N, Num features M]
            Collection of input vectors.

        Returns:
        ----------
        The net_input. Shape = [Num samples,]
        '''
        net_input = features @ self.wts + self.b
        return net_input

    def activation(self, net_in):
        '''Applies the activation function to the net input and returns the output neuron's activation.
        It is simply the identify function for vanilla ADALINE: f(x) = x

        Parameters:
        ----------
        net_in: ndarray. Shape = [Num samples N,]

        Returns:
        ----------
        net_act. ndarray. Shape = [Num samples N,]
        '''
        net_act = net_in
        return net_act

    def predict(self, features):
        '''Predicts the class of each test input sample

        Parameters:
        ----------
        features: ndarray. Shape = [Num samples N, Num features M]
            Collection of input vectors.

        Returns:
        ----------
        The predicted classes (-1 or +1) for each input feature vector. Shape = [Num samples N,]

        NOTE: Remember to apply the activation function!
        '''
        # net input --> weighted sum
        net_in = self.net_input(features)

        # net activation 
        net_act = self.activation(net_in)

        #predict
        predicted_classes = np.where(net_act >=0, 1, -1)

        #return predicted classes
        return predicted_classes

    def accuracy(self, y, y_pred):
        ''' Computes accuracy (proportion correct) (across a single training epoch)

        Parameters:
        ----------
        y: ndarray. Shape = [Num samples N,]
            True classes corresponding to each input sample in a training epoch  (coded as -1 or +1).
        y_pred: ndarray. Shape = [Num samples N,]
            Predicted classes corresponding to each input sample (coded as -1 or +1).

        Returns:
        ----------
        float. The accuracy for each input sample in the epoch. ndarray.
            Expressed as proportions in [0.0, 1.0]
        '''
        zeroes_mat = y-y_pred #zero signifies a match
        num_matches = np.count_nonzero(zeroes_mat == 0)
        acc = num_matches / y.shape[0]
        return acc

    def loss(self, y, net_act):
        ''' Computes the Sum of Squared Error (SSE) loss (over a single training epoch)

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
        loss = 0.5*(np.sum((y-net_act)**2))
        return loss

    def gradient(self, errors, features):
        ''' Computes the error gradient of the loss function (for a single epoch).
        Used for backpropogation.

        Parameters:
        ----------
        errors: ndarray. Shape = [Num samples N,]
            Difference between class and output neuron's activation value
        features: ndarray. Shape = [Num samples N, Num features M]
            Collection of input vectors.

        Returns:
        ----------
        grad_bias: float.
            Gradient with respect to the bias term
        grad_wts: ndarray. shape=(Num features N,).
            Gradient with respect to the neuron weights in the input feature layer
        '''
        #grad_bias
        grad_bias = np.sum(errors * -1)

        #grad_wts
        grad_wts = (errors @ features) * -1

        #return 
        return grad_bias, grad_wts

    def fit(self, features, y, n_epochs=1000, lr=0.001, r_seed=None):
        '''Trains the network on the input features for self.n_epochs number of epochs

        Parameters:
        ----------
        features: ndarray. Shape = [Num samples N, Num features M]
            Collection of input vectors.
        y: ndarray. Shape = [Num samples N,]
            Classes corresponding to each input sample (coded -1 or +1).
        n_epochs: int.
            Number of epochs to use for training the network
        lr: float.
            Learning rate used in weight updates during training
        r_seed: None or int.
            Random seed used for controlling the reproducability of the wts and bias

        Returns:
        ----------
        self.loss_history: Python list of network loss values for each epoch of training.
            Each loss value is the loss over a training epoch.
        self.acc_history: Python list of network accuracy values for each epoch of training
            Each accuracy value is the accuracy over a training epoch.
        '''
        #seed and gaussian dist
        rng = np.random.default_rng(r_seed)        
        # gauss = rng.normal(loc=0, scale=0.01)

        #initialize weights
        self.wts = rng.normal(loc=0, scale=0.01, size=(features.shape[1],))

        #initialize bias
        self.b = rng.normal(loc=0, scale=0.01, size=(1,))

        self.loss_history = []
        self.accuracy_history = []

        ### training loop
        for epoch in range(n_epochs):
            ## forward pass
            net_in = self.net_input(features)
            net_act = self.activation(net_in)
            predicted_classes = self.predict(features)

            ## backward pass
            #compute error
            errors = y - net_act
            # error = predicted_classes - y

            #compute loss
            loss = self.loss(y, net_act)
            self.loss_history.append(loss)

            #compute accuracy 
            accuracy = self.accuracy(y, predicted_classes)
            self.accuracy_history.append(accuracy)

            #udpate wts thru backprop
            grad_bias, grad_weights = self.gradient(errors, features) 

            self.wts -= lr * grad_weights
            self.b -= lr * grad_bias

        # return 
        return self.loss_history, self.accuracy_history
        
class Perceptron(Adaline):
    def __init__(self):
        super().__init__()
    def activation(self, net_in):
        '''Applies the activation function to the net input and returns the output neuron's activation.
        If net_in[i] >= 0, net_act[i] = 1
        If net_in[i] < 0, net_act[i] = -1

        Parameters:
        ----------
        net_in: ndarray. Shape = [Num samples N,]

        Returns:
        ----------
        net_act. ndarray. Shape = [Num samples N,]
        '''
        net_act = np.zeros(net_in.shape)
        for i in range(len(net_in)):
            if net_in[i] >= 0:
                net_act[i] = 1
            else:
                net_act[i] = -1
        return net_act
