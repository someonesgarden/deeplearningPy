#!/usr/bin/env python3
# _*_ coding:utf8 _*_

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 100
D = 2

X = np.random.randn(N,D)
X[:50, :] = X[:50, :] - 2*np.ones((50,D))
X[50:, :] = X[50:, :] + 2*np.ones((50,D))
T = np.array([0]*50 + [1]*50)
ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, X), axis=1)
#w = np.random.randn(D + 1)
w = np.array([0,4, 4])
z = Xb.dot(w)

########################
def sigmoid(z):
    return 1/(1+np.exp(-z))

def cross_entropy(T, Y):
    return - sum((T * np.log(Y) + (1-T) * np.log(1-Y)))

def cross_entropy_num(T, Y):
    E = 0
    for i in range(N):
        if T[i]==1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1-Y[i])
    return E
#############################3

learning_rate = 0.1


Y = sigmoid(z)
print(cross_entropy_num(T,Y))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.scatter(X[:,0], X[:,1], c=T, s=100, alpha=0.5)

x_axis = np.linspace(-6,6,100)
y_axis = -x_axis
ax.plot(x_axis,y_axis)

#fig.show()
fig.savefig('out/graph/logistic2.png')