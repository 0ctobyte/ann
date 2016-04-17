#!/usr/bin/env python3

# Sekhar Bhattacharya

import numpy as np

def signum(x):
    return [1 if x > 0 else (0 if x == 0 else -1)]

class Adaline:
    def __init__(self, num_weights):
        self.weights = np.random.standard_normal((num_weights, 1))
        self.bias = np.random.standard_normal();

    def train(self, training_data, l):
        x, t = training_data
        delta = np.random.standard_normal(self.weights.shape)
        e = 0.0000001
        epoch = 0

        # Supervised learning algorithm using Widrow-Hoff learning
        while np.any(delta > e):
            o = np.dot(x, self.weights) + self.bias
            error = (t - o)
            delta = l*np.dot(x.T, error)
            self.weights += delta
            self.bias += -l*np.sum(error)

            mse = np.mean(error**2)
            print(str(epoch) + ':' + str(mse))
            epoch += 1

        return (self.weights, self.bias)

    def evaluate(self, testing_data):
        x, t = testing_data
        o = np.apply_along_axis(signum, 1, np.dot(x, self.weights) + self.bias)
        r = np.sum(o==t) / t.shape[0] 

        return (o, r)

if __name__=="__main__":
    import argparse, signal
    def exit_handler(signal, frame):
        exit(0)

    # Handle ctrl-c
    signal.signal(signal.SIGINT, exit_handler)

    parser = argparse.ArgumentParser(description="A simple program to train an Adaline")
    parser.add_argument("training_data", help="Input training data file. Should be a comma separated list of normalized attributes with each line containing a single, complete input set")
    parser.add_argument("testing_data", help="Input testing data file. Should be a comma separated list of normalized attributes with each line containing a single, complete input set")
    parser.add_argument("-l", "--lrate", help="Specify the learning rate", type=float, default=0.2)
    parser.add_argument("-v", "--verbose", help="Turn on verbose output", action="store_true")
    args = parser.parse_args()

    x = []
    t = []
    with open(args.training_data, 'r') as f:
        for line in f:
            s = line.rstrip().split(';')
            x += [[float(i) for i in s[0].split(',')]]
            t += [[float(s[1])]]
    training_data = (np.array(x), np.array(t))

    x = []
    t = []
    with open(args.testing_data, 'r') as f:
        for line in f:
            s = line.rstrip().split(';')
            x += [[float(i) for i in s[0].split(',')]]
            t += [[float(s[1])]]
    testing_data = (np.array(x), np.array(t))

    # Deterministic seed
    np.random.seed(1)

    adaline = Adaline(training_data[0].shape[1])
    w, b = adaline.train(training_data, args.lrate)
    o, r = adaline.evaluate(testing_data)

    if args.verbose: 
        print((w, b))
        for oi,ti in zip(o,t):
            print(str(np.array(oi)) + '->' + str(ti))
    
    print(str(r*100) + '%') 
