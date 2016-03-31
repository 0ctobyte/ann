#!/usr/bin/env python3

# Sekhar Bhattacharya

import numpy as np
from scipy.special import expit
import argparse, signal

def step(x):
    return 1 if x > 0 else 0

def sigmoid(x):
    return expit(x)

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def back_propagate(x, t, w, b):
    # Forward propagate through the network keeping track of sums and activations after each layer
    a = [x]
    z = [x]
    for wi,bi in zip(w,b):
        z.append(np.dot(a[-1],wi) + bi)
        a.append(sigmoid(z[-1]))
    
    # Now calculate the error between the output and the training output
    # Use the error to calculate the delta needed to apply to the weights and update the weights        
    # Back propagate through the network calculating the delta needed to update the weights between layers
    delta_w = []
    delta_b = []
    delta = (t - a[-1]) * sigmoid_prime(z[-1])
    for wi,ai,zi in reversed(list(zip(w,a,z))):
        delta_w.insert(0, np.dot(ai.T, delta))
        delta_b.insert(0, np.sum(delta, 0))
        delta = np.dot(delta, wi.T) * sigmoid_prime(zi)

    return (delta_w, delta_b, np.mean((t-a[-1])**2))

def train(x, t, w, b, l, i):
    for epoch in range(i):
        delta_w, delta_b, mse = back_propagate(x, t, w, b)
        w = [wi+l*dwi for wi,dwi in zip(w, delta_w)]
        b = [bi+l*dbi for bi,dbi in zip(b, delta_b)]
        print(str(epoch) + ":" + str(mse))
    
    return (w,b)

def _train(x, t, w, b, l, e):
    epoch = 0
    mse = 1.0
    while mse > e:
        delta_w, delta_b, mse = back_propagate(x, t, w, b)
        w = [wi+l*dwi for wi,dwi in zip(w, delta_w)]
        b = [bi+l*dbi for bi,dbi in zip(b, delta_b)]
        print(str(epoch) + ":" + str(mse))
        epoch += 1

    return (w,b)

def feed_forward(x, w, b):
    # Forward propagate through the network
    a = x
    for wi,bi in zip(w, b):
        a = sigmoid(np.dot(a, wi) + bi)

    return a

def evaluate(x, t, w, b):
    a = np.argmax([feed_forward(x, w, b)], 2).T 
    r = np.sum(a==t) / t.shape[0]

    return (a, r)

  
if __name__=="__main__":
    def exit_handler(signal, frame):
        exit(0)

    # Handle ctrl-c
    signal.signal(signal.SIGINT, exit_handler)

    parser = argparse.ArgumentParser(description="Create, train and test a neural network")
    parser.add_argument("input_file", help="Input data file. Should be a comma separated list of normalized attributes with each line containing a single, complete input set")
    parser.add_argument("weight_file", help="For training, the final weights will be output to this file. For testing, the file specifies the weights to use")
    parser.add_argument("-t", "--train", nargs='+', help="Specify number of layers and neurons per layer used for training, in the form "
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
    with open(args.input_file, 'r') as f:
        for line in f:
            c = line.rstrip().split(';')
            x += [[float(i) for i in c[0].split(',')]]
            t += [[float(i) for i in c[1].split(',')]]

    x = np.array(x)
    t = np.array(t)

    if args.train != None:
        # Deterministic seed
        np.random.seed(1)
        
        # Create synapses (weights) between each layer and biases, populating them with random weight values
        w = [np.random.standard_normal((int(u),int(v))) for u,v in zip(args.train[:-1], args.train[1:])]
        b = [np.random.standard_normal((1,int(v))) for v in args.train[1:]]

        if args.error != None:
            w,b = _train(x, t, w, b, args.learning_rate, args.error)
        else:
            w,b = train(x, t, w, b, args.learning_rate, args.iterations)

        np.save(args.weight_file, (w,b))
    else:
        w,b = np.load(args.weight_file)
        a,r = evaluate(x, t, w, b)

        if args.verbose: 
            for ai,ti in zip(a,t):
                print(str(np.array(ai)) + '->' + str(ti))
        
        print(str(r*100) + '%') 

