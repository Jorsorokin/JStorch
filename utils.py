"""
series of utilities for torch model learning
"""

import numpy as np
import torch

#####################################################################
# PLOTTING FUNCTIONS
#####################################################################

def plot(model):
    # visualize the model layers as a graph
    pass


#####################################################################
# Training utility functions
#####################################################################
class NNutils():
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

    def __init__(self,model,nnType='dnn'):
        """
        establishes link to defined model and pushes to GPU if possible
        """
        self.model = model
        self.type = nnType

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model.cuda()
        else:
            self.device = torch.device('cpu')

        self._initialize_params()

    def _initialize_params(self):
        """
        Initializes all parameters in the model
        """
        self.costs = None

        if self.type is 'dnn':
            params = self.model.state_dict()
            for param in params.keys():
                if 'weight' in param:
                    torch.nn.init.normal_(params[param]) / sqrt(params[param].shape[0])
                elif 'bias' in param:
                    torch.nn.init.constant_(params[param],0)

            self.model.load_state_dict(params) # re-supply the initialized params

    def _get_optimizer(self,optimizer,alpha,regularization):
        """
        initiates the optimizing function based on user input
        """
        if optimizer is 'adadelta':
            fcn = torch.optim.Adadelta
        elif optimizer is 'adagrad':
            fcn = torch.optim.Adagrad
        elif optimizer is 'adamax':
            fcn = torch.optim.Adamax
        elif optimizer is 'adam':
            fcn = torch.optim.Adam 
        elif optimizer is 'rms':
            fcn = torch.optim.RMSprop
        elif optimizer is 'sgd':
            fcn = torch.optim.SGD 

        return fcn(self.model.parameters(),lr=alpha,weight_decay=regularization)

    def _get_loss(self,loss):
        """
        initializes the loss function
        """
        if loss is 'mse':
            return torch.nn.MSELoss
        elif loss is 'l1':
            return torch.nn.L1Loss
        elif loss is 'cross':
            return torch.nn.CrossEntropyLoss
        elif loss is 'loglikelihood':
            return torch.nn.NLLLoss
        elif loss is 'kldiv':
            return torch.nn.KLDivLoss
        elif loss is 'bce':
            return torch.nn.BCEloss
        elif loss is 'huber':
            return torch.nn.SmoothL1Loss

    def _initialize_training(self,nEpochs,optimizer,alpha,regularization,loss):
        """
        initializes optimizer, cost, and loss function for training
        """  
        self.costs = np.zeros(nEpochs)
        self.optimizer = self._get_optimizer(optimizer,alpha,regularization)
        self.loss = self._get_loss(loss)
        self.model.train() # sets model to training mode so we can update params
    
    def forget(self):
        """
        re-initializes parameters and costs (forgets learning)
        """
        self._initialize_params()

    def find_num_batches(self,N,batchSize):
        """
        determines the # of batches for training
        """
        nBatches = int(np.floor(N / batchSize))
        extra = np.mod(N,batchSize)
        batches = [batch*batchSize for batch in range(nBatches)]
        if extra > 0:
            batches.append(batches[-1]+extra)
            nBatches += 1

        return nBatches

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

        self.model.eval() # avoids updating parameters 
      
        X_tensor = self._toTensor(X)
        Yhat = self._forward(X_tensor)
        return Yhat.detach().to('cpu').numpy()

    def _forward(self,X):
        """ 
        Perform one forward pass of the model and computes the loss
        """
        self.model(X)

    def _backward(self,Y,Yhat):
        """ 
        Performs one backward pass through the network and updates the parameters
        of the model (for all learnable parameters)
        """
        self.loss(Y,Yhat)
        self.loss.backward()
        self.optimizer.step()

    def _shuffle(self,X,Y,inds):
        """
        Shuffles the data in X and Y in place to avoid memory overhead 
        """
        np.random.shuffle(inds)
        return X[inds,:],Y[inds,:]

    def _toTensor(self,X):
        """
        converts a numpy array to a torch tensor and pushes to GPU if available
        """
        X_tensor = torch.from_numpy(X).type(torch.FloatTensor)
        X_tensor.to(self.device)
        return X_tensor