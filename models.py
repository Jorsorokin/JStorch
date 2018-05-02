"""
models.py

A series of modules for facilitating neural network architectures / learning using pytorch

The modules below are wrappers around native pytorch modules, in an attempt to abstract away from the 
(slightly) lower-level building of network architectures for facilitating rapid implementation networks.

Included below are:

dNN - a simple sequential deep neural network, with arbitrary # of layers, 
      units per layer, activation functions, and loss function

rNN - a simple recurrent neural network

"""

import numpy as np
import torch
from collections import OrderedDict
from time import time

class dNN():
    def __init__(self,layers,activations,lossfcn,normalizeLayers=True):
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

            lossfcn : 
                an alias of a torch loss function (MSEloss,NLLoss, ...) used for training the network

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

        self.model = torch.nn.Sequential(model)
        self.modeldict = model
        self.lossfcn = lossfcn

        # push to GPU if available
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model.cuda()
        else:
            self.device = torch.device('cpu')

    def _initialize(self):
        """
        Initializes parameters using xavier initialization (for weights) or setting to zero (for biases and parameters for normalizations)
        """
        params = self.model.state_dict()
        for param in params.keys():
            if 'weight' in param:
                torch.nn.init.normal_(params[param])
            elif 'bias' in param:
                torch.nn.init.constant_(params[param],0)

        self.model.load_state_dict(params)

    def _forward(self,X):
        """ 
        Perform one forward pass of the model and computes the loss
        """
        return self.model(X)

    def _compute_loss(self,Y,Yhat):
        """
        Computes the loss from the predicted output Yhat and output Y
        """
        return self.lossfcn(Yhat,Y)

    def _backward(self,loss):
        """ 
        Performs one backward pass through the network and updates the parameters
        of the model (for all learnable parameters)
        """
        loss.backward()
        self.optimizer.step()

    def _shuffle(self,X,Y,inds):
        """
        Shuffles the data in X and Y in place to avoid memory overhead 
        """
        np.random.shuffle(inds)
        return X[inds,:],Y[inds,:]

    def _toTensor(self,X):
        X_tensor = torch.from_numpy(X).type(torch.FloatTensor)
        X_tensor.to(self.device)

        return X_tensor

    def fit(self,X,Y,optimizer,alpha=0.01,regularization=0,verbose=False,nEpochs=100,batchSize=128):
        """
        Trains the network on the training data X and Y, using a specific optimizer (SGD, Adam, etc.),
        learning rate alpha, L2 regularization value, and minibatch size

        Parameters:
        ----------
            X :
                a numpy array of size N x D, where N = training samples and D = size of input layer

            Y :
                a numpy array of size N x K, where N = training samples and K = size of output layer 

            optimizer :
                an alias for a torch optimizing function (i.e. torch.nn.Adam, NOT torch.nn.Adam() )

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
            Yhat :
                predicted responses of X

            costs :
                a list of costs calulated every 100 iterations

            ROC :
                a 2 x 2 numpy array, as:
                    [ [false positive %, false negative %],
                      [true positive %, true negative %] ]
        """

        # initialize the looping variables
        N,D = X.shape
        M,K = Y.shape
        assert( N == M,'# of samples in X and Y do not match' )
        shuffleInds = np.arange(N)

        nBatches = int(np.floor(N / batchSize))
        extra = np.mod(N,batchSize)
        batches = [batch*batchSize for batch in range(nBatches)]
        if extra > 0:
            batches.append(batches[-1]+extra)
            nBatches += 1

        # create our optimizing function and put model into train mode
        self.optimizer = optimizer(self.model.parameters(),lr=alpha,weight_decay=regularization)
        self.model.train()

        # convert X and Y to torch tensors
        X_tensor = self._toTensor(X)
        Y_tensor = self._toTensor(Y)

        # initialize the parameters using xavier initialization
        self._initialize()

        # loop over epochs
        self.costs = np.zeros(nEpochs)
        tic = time()
        for epoch in range(nEpochs):

            # shuffle the indices in X and Y
            X_tensor,Y_tensor = self._shuffle(X_tensor,Y_tensor,shuffleInds)

            # reset the gradients and loss
            running_cost = 0

            for batch in range(nBatches-1):
                self.optimizer.zero_grad()

                # select indices for this batch
                indicies = torch.tensor(range(batches[batch], batches[batch+1]))
                X_batch = torch.index_select(X_tensor,0,indicies)
                Y_batch = torch.index_select(Y_tensor,0,indicies)

                # forward pass
                Yhat = self._forward(X_batch)

                # compute loss
                loss = self.lossfcn(Yhat,Y_batch) 
                running_cost += loss.item()

                # backward pass
                self._backward(loss)

            # store cost
            self.costs[epoch] = running_cost / nBatches

            # print to screen if desired
            if verbose:
                print('cost is: ' + str(self.costs[epoch]))

        toc = time()
        print('Finished training after ' + str(toc-tic) + ' seconds')

    def predict(self,X):
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

        # set model to eval mode
        self.model.eval()

        X_tensor = self._toTensor(X)
        Yhat = self._forward(X_tensor)

        # convert back to numpy array
        return Yhat.detach().to('cpu').numpy()
