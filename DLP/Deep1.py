import numpy as np
import matplotlib.pyplot as plt
from lib.util import sigmoid, sigmoid_cost,softmax

N, D, M = 100, 3, 5
X = np.random.randn(N, D)

#input weight
W = np.random.randn(D, M)
#output weight
V = np.random.randn(M, D)

Z = sigmoid(X.dot(W))
A = Z.dot(V)
exp_A = np.exp(A)
exp_x = np.exp(X)

out = softmax(A)

out = exp_x / exp_x.sum(axis=1, keepdims=True)


def forward(X, W1, W2):
    #Sigmoid Z = sigma(W1*X)
    #X: input
    #Z: hidden layer
    #Y: output
    # X -- w1 -- Z --- w2 --- Y

    Z = sigmoid(X.dot(W1))
    Y = softmax(Z.dot(W2))

    return Y, Z

def derivative_w2(Z, T, Y):
    N, K = T.shape
    M = Z.shape[1]
    # ret1 = np.zeros((M,K))
    # for n in xrange(N):
    #     for m in xrange(M):
    #         for k in xrange(K):
    #             ret1[m,k] +=(T[n,k]-Y[n,k])*Z[n,m]
    ret4 = Z.T.dot(T - Y)
    # assert (np.abs(ret1 - ret4).sum() < 1e-10)
    return ret4

def derivative_w1(X, Z, T, Y, W2):
    N,D = X.shape
    M,K = W2.shape
    # ret1 = np.zeros((D,M))
    # for n in xrange(N):
    #     for k in xrange(K):
    #         for m in xrange(M):
    #             for d in xrange(D):
    #                 ret1[d,m] +=(T[n,k]-Y[n,k])*W2[m,k]*Z[n,m]*(1-Z[n,m])*X[n,d]
    ret2 = X.T.dot(((T-Y).dot(W2.T)*(Z*(1-Z))))
    # assert(np.abs(ret1 - ret2).sum() < 1e-8)
    return ret2

def cost(T, Y):
    return -(T * np.log(Y)).sum()

def main():
    N, D = 10, 3
    M = 5
    learning_rate = 1e-5
    reg = 1e-1
    epoch = 100000

    X = np.random.randn(N, D)
    W1 = np.random.randn(D, M)
    W2 = np.random.randn(M, D)
    T = np.zeros((N, D))
    LL = []

    for i in xrange(N):
        T[np.random.randint(3)] = 1

    for epoch_ in xrange(epoch):
        output, hidden = forward(X, W1, W2)
        if epoch_ % 100 == 0:
            ll=cost(T, output)
            print ll
            LL.append(ll)
        W2 += learning_rate * (derivative_w2(hidden, T, output) - reg * W2)
        W1 += learning_rate * (derivative_w1(X, hidden, T, output, W2) - reg * W1)

    plt.plot(LL)
    plt.show()

#if __name__ =='__main__':
main()
