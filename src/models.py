"""
A series of models for facilitating neural network architectures and learning using pytorch

The models below are wrappers around native pytorch modules, in an attempt to abstract away from the 
(slightly) lower-level building of network architectures and for facilitating rapid implementation of different networks.

Included below are:

DNN - a simple sequential deep neural network, with arbitrary # of layers, 
      units per layer, activation functions, and loss function

RNN - a (stacked) recurrent network, with arbitrary # of RNN layers (user defined), and an 
      option to add a final MLP output layer

Written by Jordan Sorokin

Change Log:
5/1/18 - finished DNN architecture
3/5/19 - updated class structure to inherit training utilities from NNutils superclass
3/6/19 - working on stacked RNN model architecture (perhaps make RNN its own class?)
3/7/19 - finished RNN architecture; allows for stacking and final MLP output layer
"""

import numpy as np
import torch, utils
from collections import OrderedDict
from time import time

class DNN(utils.NN):
    def __init__(self,layers,activations):
        """
        Build a deep neural network with arbitrary # of layers & units, and varied activation functions per layer

        Parameters:
        ----------
            layers :
                A tuple or list of integers of length L, where L stands for the total # of layers (including input and outputs). 
                Each element represents the # of units (or dimensions) for that layer.

                ex: layers = (12, 5, 3, 1) builds a network with 2 hidden layers with 5 and 3 units, an output layer with 
                    a single unit, and an input "layer" containing 12 dimensions

            activations :
                A list of strings of activation functions for the layers, of length L - 1. 
                Each strings must be of one of the following:
                    ['ELU','ReLU','LeakyReLU','PReLU','Sigmoid','LogSigmoid','Tanh','Softplus','Softmax','LogSoftmax']
        """
        model = utils.buildMLP(model,layers,activations)
        super().__init__(self,torch.nn.Sequential(model),type='dnn') 

    def fit(self,X,Y,loss,optimizer,alpha=0.01,regularization=0,verbose=False,nEpochs=100,batchSize=128):
        """
        Trains the network on the training data X and Y, using a specific optimizer (SGD, Adam, etc.),
        learning rate alpha, L2 regularization value, and minibatch size

        Parameters:
        ----------
            X :
                a numpy array of size N x D, where N = training samples and D = size of input layer

            Y :
                a numpy array of size N x K, where N = training samples and K = size of output layer 

            loss:
                string for the loss function, of one of the following:
                ['l2','l1','huber','kldiv','cross','llh','bce']

            optimizer :
                the type of learning optimizer, of one of the following:
                    ['adagrad', 'adadelta','adamax','adam','rms','sgd']

            alpha :
                the learning rate for the optimizer. Default = 0.01

            regularization :
                the weight decay for L2 regularization on the network. Default = 0

            verbose :
                a boolean flag for printing iterative costs to screen. Default = false

            nEpochs :
                an integer specifying the # of iterations for training. Default = 100

            batchSize :
                an integer specifying the # of training samples per batch

        Returns:
        -------
            costs :
                a list of costs calulated every 100 iterations
        """

        # check X, Y shapes and convert to tensors
        N, D_in = X.shape
        M, D_out = Y.shape
        assert( N == M,'# of samples in X and Y do not match' )
        
        shuffleInds = range(N)
        X_tensor = self._toTensor(X)
        Y_tensor = self._toTensor(Y)    
        
        # initialize the training regime and params
        self._initialize_training(nEpochs,optimizer,alpha,regularization,loss)  
        nBatches = utils.find_num_batches(N,batchSize)

        # --- epoch loop
        tic = time()
        for epoch in range(nEpochs):
            running_cost = 0
            X_tensor, Y_tensor = utils.shuffle(X_tensor,Y_tensor,shuffleInds)
            
            # --- minibatch loop 
            for batch in range(nBatches-1):
                self.optimizer.zero_grad() # resets gradients to avoid accumulation
                indicies = torch.tensor(range(batches[batch], batches[batch+1]))
                X_batch = torch.index_select(X_tensor,0,indicies) 
                Y_batch = torch.index_select(Y_tensor,0,indicies)

                # do one full pass through model
                Yhat = self._forward(X_batch)
                self._backward(Y_batch,Yhat)
                  
                running_cost += self.loss.item()

            # store cost and print to screen 
            self.costs[epoch] = running_cost / nBatches
            if verbose:
                print('cost is: ' + str(self.costs[epoch]))

        print('Elapsed training time: ' + str(time() - tic) + ' seconds')


class RNN(utils.NN):
    def __init__(self,layers,rnnType,outputDims=None,activations=None,bias=True,dropout=0.0):
        """
        Builds a (stacked) recurrent neural network with arbitrary # of stacked layers, units per layer, and rnn type per layer.
        One can also add a final MLP as an output, which can be useful for classifying time series into groups or 
        producing a forecast of future values given the RNN hidden states

        Parameters:
        ----------
            layers :
                A tuple or list of integers of length L, where L stands for the total # of layers (including input and outputs). 
                Each element represents the # of OUTPUT dimensions for that layer; thus the expected input dimensions
                sizes for each lth layer is computed as the # of output dimensions of the lth-1 layer. 

                For the first layer (input layer), specify the input & ouput dimensions as: (inputDim, outputDim)

                ex: layers = ((5,12), 5, 3, 1) builds a recurrent network with 4 stacked layers with the following I/O dimensions:
                                    
                                    input dims      output dims
                        layer 1:        5               12
                        layer 2:        12              5
                        layer 3:        5               3
                        layer 4:        3               1

            rnnType :
                A list of strings representing the rnn cell type for each layer in a stacked RNN (if layer > 1)
                    'RNN' - vanilla recurrent network cell
                    'LSTM' - long short-term memory cell
                    'GRU' - gated recurrent unit cell
            
            outputDims :
                List of integers. If not None, the final output layer of the network is an MLP, with the # of 
                inputs equal to the # of dims of the final RNN layer, and the # of outputs equal to outputDim. 
                Default = None

            activations :
                A list of strings for the output MLP activation functions (only used if outputDim > 0).

                Each string must be of one of the following:
                    ['ELU','ReLU','LeakyReLU','PReLU','Sigmoid','LogSigmoid','Tanh','Softplus','Softmax','LogSoftmax']

                Default = None

            bias :
                a boolean flag for computing bias from the network layers
                Default = True

            dropout :
                float between [0,1] indicating the probability of dropout for regularizing the RNN. Any float > 0.0 
                results in a dropout layer following each RNN layer, with P(dropout_i) = dropout 
                Default = 0.0
        """
        model = buildRNN(layers, rnnType, dropout)

        if outputDims is not None:
            layers = [layers[-1]].append(outputDims)
            model.update(utils.buildMLP(layers,activations))

        super().__init__(self,torch.nn.Sequential(model),type='rnn')

    def fit(self,X,Y,loss,optimizer,alpha=0.01,regularization=0,verbose=False,nEpochs=100,batchSize=128):
        """
        Trains the network on the training data X and Y, using a specific optimizer (SGD, Adam, etc.),
        learning rate alpha, L2 regularization value, and minibatch size

        Parameters:
        ----------
            X :
                a numpy array of size N x D, where N = training samples and D = size of input layer

            Y :
                a numpy array of size N x K, where N = training samples and K = size of output layer 

            loss:
                string for the loss function, of one of the following:
                ['l2','l1','huber','kldiv','cross','llh','bce']

            optimizer :
                the type of learning optimizer, of one of the following:
                    ['adagrad', 'adadelta','adamax','adam','rms','sgd']

            alpha :
                the learning rate for the optimizer. Default = 0.01

            regularization :
                the weight decay for L2 regularization on the network. Default = 0

            verbose :
                a boolean flag for printing iterative costs to screen. Default = false

            nEpochs :
                an integer specifying the # of iterations for training. Default = 100

            batchSize :
                an integer specifying the # of training samples per batch

        Returns:
        -------
            costs :
                a list of costs calulated every 100 iterations
        """

        # check X, Y shapes and convert to tensors
        N, D_in = X.shape
        M, D_out = Y.shape
        assert( N == M,'# of samples in X and Y do not match' )
        
        shuffleInds = range(N)
        X_tensor = self._toTensor(X)
        Y_tensor = self._toTensor(Y)    
        
        # initialize the training regime and params
        self._initialize_training(nEpochs,optimizer,alpha,regularization,loss)  
        nBatches = utils.find_num_batches(N,batchSize)

        # --- epoch loop
        tic = time()
        for epoch in range(nEpochs):
            running_cost = 0
            X_tensor, Y_tensor = utils.shuffle(X_tensor,Y_tensor,shuffleInds)
            
            # --- minibatch loop 
            for batch in range(nBatches-1):
                self.optimizer.zero_grad() # resets gradients to avoid accumulation
                indicies = torch.tensor(range(batches[batch], batches[batch+1]))
                X_batch = torch.index_select(X_tensor,0,indicies) 
                Y_batch = torch.index_select(Y_tensor,0,indicies)

                # do one full pass through model
                Yhat = self._forward(X_batch)
                self._backward(Y_batch,Yhat)
                  
                running_cost += self.loss.item()

            # store cost and print to screen 
            self.costs[epoch] = running_cost / nBatches
            if verbose:
                print('cost is: ' + str(self.costs[epoch]))

        print('Elapsed training time: ' + str(time() - tic) + ' seconds')
