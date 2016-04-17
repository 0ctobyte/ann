#!/usr/bin/env python3

# Sekhar Bhattacharya

import numpy as np
from scipy.special import expit

def step(x):
    return 1 if x > 0 else 0

def sigmoid(x):
    return expit(x)

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

class MLPerceptron:
    def __init__(self, layers):
        # Create synapses (weights) between each layer and biases, populating them with random weight values
        self.weights = [np.random.standard_normal((u,v)) for u,v in zip(layers[:-1], layers[1:])]
        self.bias = [np.random.standard_normal((1,v)) for v in layers[1:]]

    def back_propagate(self, x, t):
        # Forward propagate through the network keeping track of sums and activations after each layer
        a = [x]
        z = [x]
        for wi,bi in zip(self.weights, self.bias):
            z.append(np.dot(a[-1], wi) + bi)
            a.append(sigmoid(z[-1]))
        
        # Now calculate the error between the output and the training output
        # Use the error to calculate the delta needed to apply to the weights and update the weights        
        # Back propagate through the network calculating the delta needed to update the weights between layers
        delta_w = []
        delta_b = []
        delta = (t - a[-1]) * sigmoid_prime(z[-1])
        for wi,ai,zi in reversed(list(zip(self.weights, a, z))):
            delta_w.insert(0, np.dot(ai.T, delta))
            delta_b.insert(0, np.sum(delta, 0))
            delta = np.dot(delta, wi.T) * sigmoid_prime(zi)
    
        return (delta_w, delta_b, np.mean((t-a[-1])**2))

    def train(self, training_data, l, i):
        x, t = training_data
        for epoch in range(i):
            delta_w, delta_b, mse = self.back_propagate(x, t)
            self.weights = [wi+l*dwi for wi,dwi in zip(self.weights, delta_w)]
            self.bias = [bi+l*dbi for bi,dbi in zip(self.bias, delta_b)]
            print(str(epoch) + ":" + str(mse))
        
        return (self.weights, self.bias)

    def train2(self, training_data, l, e):
        x, t = training_data
        epoch = 0
        mse = 1.0
        while mse > e:
            delta_w, delta_b, mse = self.back_propagate(x, t)
            self.weights = [wi+l*dwi for wi,dwi in zip(self.weights, delta_w)]
            self.bias = [bi+l*dbi for bi,dbi in zip(self.bias, delta_b)]
            print(str(epoch) + ":" + str(mse))
            epoch += 1
    
        return (self.weights, self.bias)

    def feed_forward(self, x):
        # Forward propagate through the network
        a = x
        for wi,bi in zip(self.weights, self.bias):
            a = sigmoid(np.dot(a, wi) + bi)
    
        return a
    
    def evaluate(self, testing_data):
        x, t = testing_data
        a = np.argmax([self.feed_forward(x)], 2).T 
        r = np.sum(a==t) / t.shape[0]
    
        return (a, r)

  
if __name__=="__main__":
    import argparse, signal
    def exit_handler(signal, frame):
        exit(0)

    # Handle ctrl-c
    signal.signal(signal.SIGINT, exit_handler)

    parser = argparse.ArgumentParser(description="Create, train and test a neural network")
    parser.add_argument("training_data", help="Training data file. Should be a comma separated list of normalized attributes with each line containing a single, complete input set")
    parser.add_argument("testing_data", help="Testing data file. Should be a comma separated list of normalized attributes with each line containing a single, complete input set")
    parser.add_argument("train", nargs='+', help="Specify number of layers and neurons per layer used for training, in the form "
                                       "'j0 ... jn' where 'j0' is the number of neurons for the input layer, "
                                       "'j1' to 'j(n-1)' are the number of neurons for each of the n hidden layers "
                                       "and 'jn' is the number of output neurons")
    parser.add_argument("-i", "--iterations", help="Number of iterations for training", type=int, default=10000)
    parser.add_argument("-e", "--error", help="Train until error is less than this value", type=float)
    parser.add_argument("-l", "--learning_rate", help="Specify learning rate to modulate back propagation weight update", type=float, default=0.2)
    parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true")
    args = parser.parse_args()

    # Collect the test/train inputs and test/train outputs
    x = []
    t = []
    with open(args.training_data, 'r') as f:
        for line in f:
            c = line.rstrip().split(';')
            x += [[float(i) for i in c[0].split(',')]]
            t += [[float(i) for i in c[1].split(',')]]
    training_data = (np.array(x), np.array(t))

    x = []
    t = []
    with open(args.testing_data, 'r') as f:
        for line in f:
            c = line.rstrip().split(';')
            x += [[float(i) for i in c[0].split(',')]]
            t += [[float(i) for i in c[1].split(',')]]
    testing_data = (np.array(x), np.array(t))

    # Deterministic seed
    np.random.seed(1)
   
    layers = [int(i) for i in args.train]
    mlp = MLPerceptron(layers)
    
    if args.error != None:
        w, b = mlp.train2(training_data, args.learning_rate, args.error)
    else:
        w, b = mlp.train(training_data, args.learning_rate, args.iterations)

    a, r = mlp.evaluate(testing_data)

    if args.verbose: 
        print((w, b))
        for ai,ti in zip(a,t):
            print(str(np.array(ai)) + '->' + str(ti))
    
    print(str(r*100) + '%') 
