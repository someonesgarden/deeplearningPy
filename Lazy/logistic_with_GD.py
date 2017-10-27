#!/usr/bin/env python3
#_*_ coding_utf8 _*_
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from process import get_binary_data

N, D = 100, 2

def sigmoid(z):
    return 1/(1+np.exp(-z))

def forward(A,W,b):
    return sigmoid(A.dot(W) + b)

def classification_rate(Y, P):
    return np.mean(Y == P)

# cross entropy
def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY) + (1 - T)*np.log(1 - pY))


# def cross_entropy(T, Y):
#     E = 0
#     for i in range(N):
#         if T[i] == 1:
#             E -=np.log(Y[i])
#         else:
#             E -= np.log(1-Y[i])
#     return E

#################################

X, Y = get_binary_data()
X, Y = shuffle(X, Y)
Xtrain = X[:-100]
Ytrain = Y[:-100]
Xtest  = X[-100:]
Ytest  = Y[-100:]


#shape
# X:(N,D)
# W:(D,)
# X.dot(W):(N,)
D, N = X.shape[1], X.shape[0]
W = np.random.randn(D)
b = 0


train_costs = []
test_costs = []
learning_rate = 0.001
EPOCH = 10000

for i in range(EPOCH):
    pYtrain = forward(Xtrain, W, b)
    pYtest  = forward(Xtest,  W, b)
    ctrain  = cross_entropy(Ytrain, pYtrain)
    ctest   = cross_entropy(Ytest, pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)

    W -= learning_rate*Xtrain.T.dot(pYtrain - Ytrain)
    b -= learning_rate*(pYtrain - pYtrain).sum()

    if i % 1000 ==0:
        print(i, ctrain, ctest)


print("Final train classification_rate:", classification_rate(Ytrain, np.round(pYtrain)))
print("Final test classification_rate:", classification_rate(Ytest, np.round(pYtest)))


fig = plt.figure()
ax = fig.add_subplot(1,1,1)


legend1, = ax.plot(train_costs, label='train cost')
legend2, = ax.plot(test_costs, label='test cost')
ax.legend([legend1, legend2])

fig.savefig('out/graph/logistic_with_GD.png')