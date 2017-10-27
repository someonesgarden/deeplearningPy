import numpy as np
import matplotlib.pyplot as plt
from lib.util import sigmoid_cost, sigmoid

N, D = 4, 2
epoch, learning_rate = 5000, 1e-3

X = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
)

#target
T = np.array([0, 1, 1, 0])

ones = np.array([[1]* N ]).T
xy = np.matrix(X[:, 0] * X[:, 1]).T
Xb = np.array(np.concatenate((ones, xy, X), axis = 1))

w = np.random.randn(D + 2)
z = Xb.dot(w)
Y = sigmoid(z)

error = []

for i in xrange(epoch):
    e = sigmoid_cost(T, Y)
    error.append(e)
    if i % 100 == 0:
        print e

    w += learning_rate * (np.dot((T - Y).T, Xb) - 0.01 * w)
    Y = sigmoid(Xb.dot(w))

plt.plot(error)
plt.title("Cross-entropy per iteration")
plt.show()

print "Final w:", w
print "Final classification rate:", 1 - np.abs(T - np.round(Y)).sum() / N

