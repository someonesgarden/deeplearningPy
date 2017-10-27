import numpy as np
from lib.util import sigmoid, sigmoid_cost, cross_entropy
from lib.plots import showScatter

N, D = 100, 2
EPOCH = 200
learning_rate = 1e-1

X = np.random.randn(N, D)
# center the first 50 points at (-2,-2)
X[:50, :] = X[:50, :] - 2*np.ones((50, D))
# center the last 50 points at (2,2)
X[50:, :] = X[50:, :] + 2*np.ones((50, D))
#target labels
T = np.array([0]*50 + [1]*50)


I = np.array([[1]*N]).T
Xb = np.concatenate((I, X), axis=1)
w = np.random.randn(D + 1)
z = Xb.dot(w)
Y = sigmoid(z)

for i in xrange(EPOCH):
    if i % 10 ==0:
        print sigmoid_cost(T, Y)

    #gradient descent weight update
    w += learning_rate*np.dot((T-Y).T, Xb)

    # recalculate Y
    Y = sigmoid(Xb.dot(w))

print "Final w:", w


