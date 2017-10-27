#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import sys, os
sys.path.append(os.pardir)

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from basic.util import get_normalized_data,y2indicator


def error_rate(p, t):
    return np.mean(p != t)

def relu(a):
    return a * (a>0)


def main():
    X, Y = get_normalized_data()

    max_iter = 20
    print_period = 10
    lr = 0.00004
    reg = 0.01

    Xtrain = X[:-1000,]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:,]
    Ytest = Y[-1000:]
    print(Xtrain)




if __name__ =='__main__':
    main()