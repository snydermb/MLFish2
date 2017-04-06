import numpy as np
from math import pow

## John Santaguida and Malcolm Snyder primary neural net framework. Note - could
## not get implementation of mini-batches to run, so those related lines of code
## have been excluded for the scope of this assignment. Additionally, while our
## Network will function with any given number of input units, output units, and
## hidden layers, hidden layer SIZE must remain fixed for the time being.
class Network(object):

    ## Notice that our inputs and outputs are processed here first -
    ## for an example of our main driver code to run this implementation,
    ## see driver.py
    def __init__(self, inputs, outputs):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        np.random.seed(1)

    ## Where the "learning" of our network takes place, by constantly shifting
    ## the values in w1 and w2. Backpropogation has been implemented here, as
    ## well as feeding forward. The only remaining piece to implement in the
    ## future will be the mini-batches.
    def SGD(self, numLayers, alpha, epochs):
        x = self.inputs.shape[1]
        biases = []
        for n in range(x):
            biases.append(np.random.randn(x,1))
        w0 = 2*np.random.random((x,numLayers)) - 1
        w1 = 2*np.random.random((numLayers,1)) - 1
        l0 = self.inputs
        for iter in range(epochs):
            l1 = sigmoid(np.dot(l0, w0))
            l2 = sigmoid(np.dot(l1, w1))
            l2_err = simpleDifference(l2, self.outputs)
            ##if (iter%2500) == 0:
            ##    print "Cost:" + str(np.mean(np.abs(l2_err)))
            l2_delta = l2_err*deriv(l2)
            l1_err = l2_delta.dot(w1.T)
            l1_delta = l1_err*deriv(l1)
            w1 -= alpha * l1.T.dot(l2_delta)
            ##if (iter%2500) == 0:
            ##    print "Weights:" + str(w1)
            w0 -= alpha * l0.T.dot(l1_delta)

# # # # # # # #

## Straightforward sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

## Helper function which returns the derivative value(s) of a component
def deriv(z):
    return z*(1-z)

## Basic cost function calculating merely a direct difference.
def simpleDifference(x1, x2):
    return (x1-x2)

## Quadratic cost function computation as defined in the Nielsen reading,
## chapter 2. Note - could not get this implementation to work with the rest
## of code, so for the purposes of the network we relied mainly on the
## basic simpleDifference cost function.
def costFn(result, actual):
    err = np.abs(simpleDifference(result, actual))
    err = np.power(err, 2)
    return np.sum(err)
