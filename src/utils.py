"""
series of utilities for torch model learning
"""

import numpy as np
import torch
from torch import nn, optim
from collections import OrderedDict


lossFcns = {
    'l2': nn.MSELoss,
    'l1': nn.L1Loss,
    'cross': nn.CrossEntropyLoss,
    'llh': nn.NLLLoss,
    'kldiv': nn.KLDivLoss,
    'bce': nn.BCELoss,
    'huber': nn.SmoothL1Loss
}

optimFcns = {
    'adadelta': optim.Adadelta, 
    'adagrad': optim.Adagrad,
    'adamax': optim.Adamax,
    'adam': optim.Adam,
    'rms': optim.RMSprop,
    'sgd': optim.SGD 
}


#####################################################################
# PLOTTING FUNCTIONS
#####################################################################
def plot(model):
    # visualize the model layers as a graph
    pass

#####################################################################
# UTILITY FUNCTIONS
#####################################################################
def buildMLP(layers, activations):
    """
    builds a dense MLP using the provided activation functions and layer sizes

    Parameters:
    ----------
        layers :
            list or tuple of ints specifying the sizes of the layers for the network, 
            including the "input" layer (the expected size of the input)

        activations :
            list of activation functions for each layer (see documentation under models.DNN)
    
    Returns:
    -------
        model :
            an OrderedDict() containing initiated Linear layers. All layers minus the output layer will be 
            normalized via BatchNorm1d
    """

    model = OrderedDict()
    nLayers = len(layers)
    for l in range(nLayers-1):
        model['linear_' + str(l+1)] = torch.nn.Linear(layers[l],layers[l+1])
        model['activation_' + str(l+1)] = getattr(torch.nn,activations[l])()
        if l < nLayers-2:
            model['norm_' + str(l+1)] = torch.nn.BatchNorm1d(layers[l+1])

    return model

def buildRNN(layers,cellType,dropout):
    """
    builds a stacked recurrent neural network with user-specifying RNN cells at each layer

    Parameters:
    ----------
        layers :
            list or tuple of ints specifying the sizes of the layers for the network, 
            including the "input" layer (the expected size of the input)

        cellTypes :
            a list of strings specifying the the types of recurrent networks for each layer
            Must be either 'RNN', 'LSTM', or 'GRU'

        dropout :
            a fload between [0,1] specifying dropout probability for the RNN layers. 
            If dropout = 0, no dropout layer will be added
    """
    model = OrderedDict()
    L = len(layers)

    for l in range(L):
        if l == 0:
            inSize, outSize = layers[l]
        else:
            inSize = layers[l-1]
            outSize = layers[l]
            if type(inSize) is tuple or type(inSize) is list:
                inSize = inSize[-1]

        model['recurrent_' + str(l+1)] = getattr(torch.nn, cellType[l])(inSize, outSize)
        if l < L and dropout > 0:
            model['dropout_' + str(l+1)] = torch.nn.Dropout(dropout)

    return model

def set_initial_conditions(model):
    """
    initializes all weights to random normal and biases to 0

    Params:
    ------
        model :
            a torch.nn.Sequential model
    """
    params = model.state_dict()
    for param in params.keys():    
        if 'weight' in param:
            torch.nn.init.normal_(params[param]) / np.sqrt(params[param].shape[0])
        elif 'bias' in param:
            torch.nn.init.constant_(params[param],0)

    model.load_state_dict(params)
    return model

def find_num_batches(N, batchSize):
    """
    determines the # of batches for training

    Parameters:
    ----------
        N :
            integer, the # of training samples

        batchSize :
            integer, the desired size of each training batch
    """
    nBatches = int(np.ceil(N / batchSize))
    batches = [batch*batchSize for batch in range(nBatches)]
    batches[-1] = N

    return nBatches, batches

def shuffle(X, Y, inds):
    """
    Shuffles the data in X and Y in place to avoid memory overhead 
    """
    np.random.shuffle(inds)
    return X[inds,:],Y[inds,:]


#####################################################################
# GENERIC NETWORK CLASS
#####################################################################
class NN:
    """
    A generic class containing initialization and training methods used across various
    neural network architectures (deep sequential, recurrent, ...) in "model.py". The methods are 
    useless unless instantiated as a superclass of a specified network architecture, since
    the actual learning updates will be customized for various architectures.

    Parameters:
    ----------
        model : 
            an initialized model (i.e. torch.nn.Sequential(model_params))

    Returns:
    -------
        initialized parameters and generic learning methods 
    """
    def __init__(self, model, nnType):
        """
        establishes link to defined model and pushes to GPU if possible
        """

        self.model = model
        self.type = nnType

        self._initialize_params()

    def _initialize_params(self):
        """
        Initializes all parameters in the model
        """
        self.model = set_initial_conditions(self.model)
        self._push_to_cuda()
        self.costs = None

    def _initialize_training(self, nEpochs, optimizer, alpha, regularization, loss):
        """
        initializes optimizer, cost, and loss function for training
        """  
        self.costs = np.zeros(nEpochs)
        self.optimizer = optimFcns[optimizer](
            self.model.parameters(), lr=alpha, weight_decay=regularization
        )
        self.optimizer = self._get_optimizer(optimizer, alpha, regularization)
        self.loss = lossFcns[loss]()
        self.model.train() # sets model to training mode so we can update params

    def _push_to_cuda(self):
        """
        allows for GPU computation if GPU available
        """
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model.cuda()
        else:
            self.device = torch.device('cpu')

    def _get_optimizer(self, optimizer, alpha, regularization):
        """
        initiates the optimizing function
        """
        fcn = optimFcns[optimizer]
        return fcn(self.model.parameters(), lr=alpha, weight_decay=regularization)

    def _forward(self, X):
        """ 
        Perform one forward pass of the model and computes the loss
        """
        self.model(X)

    def _backward(self, Y, Yhat):
        """ 
        Performs one backward pass through the network and updates the parameters
        of the model (for all learnable parameters)
        """
        loss = self.loss(Y, Yhat)
        loss.backward()
        self.optimizer.step()

    def _toTensor(self, X):
        """
        converts a numpy array to a torch tensor and pushes to GPU if available
        """
        X_tensor = torch.from_numpy(X).type(torch.FloatTensor)
        X_tensor.to(self.device)
        return X_tensor


    def forget(self):
        """
        re-initializes parameters and costs (forgets learning)
        """
        self._initialize_params()

    def update(self, model):
        """
        updates the torch model in "self.model" by appending a new model
        as the last "layer" in the current model
        """
        for layer in model.named_children():
            self.model.add_module(layer[0], layer[1])

    def predict(self, X):
        """
        Predict the response to X from a forward pass through the network

        Parameters:
        ----------
            X :
                a numpy array of size M x D, where M = samples and D = size of input layer

        Returns:
        -------
            Yhat :
                predicted responses of X of size M x K, where K = size of output layer
        """

        self.model.eval() # avoids updating parameters 
      
        X_tensor = self._toTensor(X)
        Yhat = self._forward(X_tensor)
        return Yhat.detach().to('cpu').numpy()

    def print(self):
        print('network architecture:\n')
        print('-----------------------')
        print(self.model)
        print('-----------------------\n')