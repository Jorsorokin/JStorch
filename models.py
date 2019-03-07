"""
A series of models for facilitating neural network architectures and learning using pytorch

The models below are wrappers around native pytorch modules, in an attempt to abstract away from the 
(slightly) lower-level building of network architectures and for facilitating rapid implementation of different networks.

Included below are:

dNN - a simple sequential deep neural network, with arbitrary # of layers, 
      units per layer, activation functions, and loss function


Written by Jordan Sorokin

Change Log:
5/1/18 - finished DNN architecture
3/5/19 - updated class structure to inherit training utilities from NNutils superclass
3/6/19 - working on stacked RNN model architecture (perhaps make RNN its own class?)
"""

import numpy as np
import torch, utils
from collections import OrderedDict
from time import time
from utils import NNutils

class DNN(NNutils):
    def __init__(self,layers,activations,normalizeLayers=True):
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
                A list of aliased activation functions for the layers, of length L - 1, since we don't apply activations 
                at the input layer. Valid activations are any contained in torch.nn.

            normalizeLayers :
                a boolean flag for normalizing hidden layers. If true, each layer will be normalized via batch normalization.
                Default = True
        """
        self.L = len(layers)
        model = OrderedDict()
        for l in range(1,self.L):
            model['linear' + str(l)] = torch.nn.Linear(layers[l-1],layers[l])
            model['activation' + str(l)] = activations[l-1] # l-1 since input layer does not have an activation
            if normalizeLayers and l != self.L-1:
                model['normalization' + str(l)] = torch.nn.BatchNorm1d(layers[l])

        # initialize model and parameters
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
                ['mse','l1','huber','kldiv','cross','loglikelihood','bce']

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
        nBatches = self.find_num_batches(N,batchSize)

        # --- epoch loop
        tic = time()
        for epoch in range(nEpochs):
            running_cost = 0
            X_tensor, Y_tensor = self._shuffle(X_tensor,Y_tensor,shuffleInds)
            
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


class sRNN(NNutils):
    def __init__(self,layers,activations,bias=True,droppout=0.0):
        """
        Builds a (stacked) recurrent neural network with arbitrary # of stacked layers, units per layer, and activation functions per layer

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

            activations :
                A list of strings representing the rnn cell type for each layer in a stacked RNN (if layer > 1)
                    'rnn' - vanilla recurrent network cell
                    'lstm' - long short-term memory cell
                    'gru' - gated recurrent unit cell

                Default = ['lstm'] * len(layers)

            bias :
                a boolean flag for computing bias from the network layers
                Default = true

            droppout :
                float between [0,1] indicating the probability of droppout for regularizing the RNN
                Default = 0.0
        """
        self.L = len(layers)
        model = OrderedDict()
        for l in range(self.L):
            if l == 0:
                inSize, outSize = layers[l]
            else:
                inSize = layers[l-1][-1]
                outSize = layers[l]

            if activations[l] is 'rnn':
                model[activations[l] + str(l+1)] = torch.nn.RNN(inSize,outSize,1)
            elif activations[l] is 'lstm':
                model[activations[l] + str(l+1)] = torch.nn.LSTM(inSize,outSize,1)
            elif activations[l] is 'gru':
                model[activations[l] + str(l+1)] = torch.nn.GRU(inSize,outSize,1)

            #model['linear' + str(l+1)] = torch.nn.Linear(layers[l-1],layers[l])
            #model['activation' + str(l+1)] = activations[l-1] # l-1 since input layer does not have an activation
            
            #if normalizeLayers and l != self.L-1:
            #    model['normalization' + str(l)] = torch.nn.BatchNorm1d(layers[l])

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
                ['mse','l1','huber','kldiv','cross','loglikelihood','bce']

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
        nBatches = self.find_num_batches(N,batchSize)

        # --- epoch loop
        tic = time()
        for epoch in range(nEpochs):
            running_cost = 0
            X_tensor, Y_tensor = self._shuffle(X_tensor,Y_tensor,shuffleInds)
            
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
