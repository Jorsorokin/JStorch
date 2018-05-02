# JStorch
repository for abstracting details away from pytorch API for a simpler interface to set up NNs

JStorch provides even higher-level abstraction for building networks via pytorch than the built-in .nn library. The intended use is for quickly prototyping network architectures without worrying about lower-level details such as converting inputs/parameters to cuda arrays, ensuring batch normalization layers receive the correct # of inputs, etc.

The models are contained in JStorch.models, and are designed to be modular. Currently models include:

[x] dNN - a modular architecture for building a sequential deep NN with 1D inputs. 
