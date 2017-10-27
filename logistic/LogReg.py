import numpy as np
from lib.util import sigmoid, sigmoid_cost, cross_entropy
from lib.plots import showScatter

N, D = 100, 2
X = np.random.randn(N, D)
learning_rate = 1e-1
lambdavalue = 1e-1

#center the first 50 points at (-2, -2)
#center the last 50 points at (2, 2)
#X[:50, :] = X[:50, :] + np.array([[-2,-2]]*50)
#X[50:, :] = X[50:, :] + np.array([[2, 2]]*50)
X[:50, :] = X[:50, :] - 2*np.ones((50, D))
X[50:, :] = X[50:, :] + 2*np.ones((50, D))
T = np.array([0]*50 + [1]*50)

ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, X), axis=1)
w = np.random.randn(D + 1)
z = Xb.dot(w)
Y = sigmoid(z)


for i in xrange(T.size):
    if i % 10 == 0:
        print sigmoid_cost(T, Y)
    w += learning_rate*(np.dot((T-Y).T, Xb) - lambdavalue * w)
    Y = sigmoid((Xb.dot(w)))

print "Final w:", w
