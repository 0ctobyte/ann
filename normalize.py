#!/usr/bin/env python3

# Sekhar Bhattacharya

import argparse

def normalize_training(f, o, m, tidx):
    for line in f:
        l = [float(i) for i in line.rstrip().split(',')]
        z = zip(l[1:],m[0][1:],m[1][1:]) if tidx == 0 else zip(l[:-1],m[0][:-1],m[1][:-1])
        x = [(i - m0i) / (m1i - m0i) for i,m0i,m1i in z]
        t = [1 if i == int(l[tidx]-m[0][tidx]) else 0 for i in range(0, int(m[1][tidx]-m[0][tidx]+1))]
        o.write(','.join([str(i) for i in x]) + ';' + ','.join([str(i) for i in t]) + '\n')

def normalize_testing(f, o, m, tidx):
    for line in f:
        l = [float(i) for i in line.rstrip().split(',')]
        z = zip(l[1:],m[0][1:],m[1][1:]) if tidx == 0 else zip(l[:-1],m[0][:-1],m[1][:-1])
        x = [(i - m0i) / (m1i - m0i) for i,m0i,m1i in z]
        o.write(','.join([str(i) for i in x]) + ';' + str(l[tidx]-m[0][tidx]) + '\n')

def normalize(f, o, m):
    for line in f:
        l = [float(i) for i in line.rstrip().split(',')]
        x = [(i - m0i) / m1i for i,m0i,m1i in zip(l,m[0],m[1])]
        o.write(','.join([str(i) for i in x]) + '\n')

def find_bounds(f, b):
    mins = [float(i) for i in f.readline().rstrip().split(',')]
    maxs = mins
    for line in f:
        l = [float(i) for i in line.rstrip().split(',')]
        mins = [min(li,mi) for li,mi in zip(l,mins)]
        maxs = [max(li,mi) for li,mi in zip(l,maxs)]

    b.write(','.join([str(i) for i in mins]) + '\n')
    b.write(','.join([str(i) for i in maxs]) + '\n')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input data file to normalize")
    parser.add_argument("output_file", help="Name of file to write normalized data")
    parser.add_argument("tidx", help="Indicates which index, in the input_file and/or bounds_file, the t value is located. "
                                     "0 indicates that 't' is in first position of each row, -1 indicates that 't' is is last position of each row", type=int, choices=[0,-1])
    parser.add_argument("-b", "--bounds", help="Input data file specifying min/max bounds. First row should be min, second row should be max of each attribute (including result value)")
    parser.add_argument("-c", "--calculate", help="Use the input_file data to create max/min bounds file", action="store_true")
    parser.add_argument("-t", "--training", help="Indicates that input file is training data", action="store_true")
    parser.add_argument("-tt", "--testing", help="Indicates that input file is testing data", action="store_true")
    args = parser.parse_args()
    
    with open(args.input_file, 'r') as f, open(args.output_file, 'w') as o: 
        if args.calculate:
            find_bounds(f, o)
        else:
            minimums = []
            maximums = []
            if args.bounds != None:
                with open(args.bounds, 'r') as b:
                    minimums = [float(i) for i in b.readline().rstrip().split(',')]
                    maximums = [float(i) for i in b.readline().rstrip().split(',')]
            else:
                minimums = [float(i) for i in f.readline().rstrip().split(',')]
                maximums = [float(i) for i in f.readline().rstrip().split(',')]

            if args.training:
                normalize_training(f, o, (minimums, maximums), args.tidx)
            elif args.testing:
                normalize_testing(f, o, (minimums, maximums), args.tidx)
            else:
                normalize(f, o (minimums[1:], maximums[1:]) if args.tidx == 0 else (minimums[:-1], maximums[:-1]))

