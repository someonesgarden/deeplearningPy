#!/usr/bin/env python3
# _*_ coding:utf8 _*_

import numpy as np
import pandas as pd
from  logistic_process import get_binary_data


X, Y = get_binary_data()


D = X.shape[1]
W = np.random.randn(D)
b = 0

def sigmoid(a):
    return 1/(1 + np.exp(-a))

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

P_Y_given_X = forward(X,W,b)

predictions = np.round(P_Y_given_X)

def classication_rate(Y, P):
    return np.mean(Y==P)

print("Score:", classication_rate(Y, predictions))

