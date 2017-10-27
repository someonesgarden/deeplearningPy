import numpy as np
import matplotlib.pyplot as plt
from lib.util import sigmoid, sigmoid_cost

N, D = 1000, 2
epoch, learning_rate = 5000, 1e-4
R_inner, R_outer = 5, 10

R1 = np.random.randn(N/2) + R_inner
theta = 2*np.pi * np.random.random(N/2)
X_inner = np.concatenate(
    [
        [R1 * np.cos(theta)],
        [R1 * np.sin(theta)]
    ]
).T

R2 = np.random.randn(N/2) + R_outer
theta = 2*np.pi * np.random.random(N/2)
X_outer = np.concatenate(
    [
        [R2 * np.cos(theta)],
        [R2 * np.sin(theta)]
    ]
).T

X = np.concatenate([X_inner, X_outer])
T = np.array([0]*(N/2) + [1]*(N/2))

plt.scatter(X[:,0], X[:,1], c=T)
plt.show()

ones = np.array([[1]*N]).T
r = np.zeros((N,1))
for i in xrange(N):
    r[i] = np.sqrt(X[i,:].dot(X[i,]))

Xb = np.concatenate((ones, r, X), axis=1)
w = np.random.randn(D + 2)
z = Xb.dot(w)
Y = sigmoid(z)

error = []

for i in xrange(epoch):
    e = sigmoid_cost(T, Y)
    error.append(e)
    if i % 100 == 0:
        print e
    w += learning_rate * (np.dot((T - Y).T, Xb) - 0.01*w)
    Y = sigmoid(Xb.dot(w))

print "Final w:",
print "Final classification rate:", 1 - np.abs(T - np.round(Y)).sum() / N


