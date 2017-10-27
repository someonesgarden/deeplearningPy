import numpy as np
import matplotlib.pyplot as plt


M = 10
def y(x, w, M):
    Y = np.array([w[i] * (x ** i) for i in xrange(M+1)])
    return Y.sum()

def E(x, t, M):
    A = np.zeros((M+1,M+1))
    for i in xrange(M+1):
        for j in xrange(M+1):
            A[i,j] = (x ** (i+j)).sum()
    T = np.array([((x**i)*t).sum() for i in xrange (M+1)])
    return np.linalg.solve(A,T)

def Erms(y,t,N):
    return np.sqrt(((y - t)**2).sum()/N)

def main():
    x_real = np.arange(0, 1, 0.01)
    y_real = np.sin(2*np.pi*x_real)
    N = 10
    loc = 0
    scale =0.3
    x_train = np.arange(0, 1, 0.1)
    y_train = np.sin(2*np.pi*x_train) + np.random.normal(loc, scale, N)
    M_sequence = np.array([0,1,2,3,4,5,6,7,8,9])
    Erms_ary = []

    for M in xrange(len(M_sequence)):
        W = E(x_train, y_train, M)
        #print W
        y_estimate = [y(x, W, M) for x in x_real]
        erms = Erms(y_estimate, y_real, N)
        print "erms"
        print erms
        Erms_ary.append(erms)

        plt.plot(x_real, y_estimate, 'r-')
        plt.plot(x_train, y_train, 'bo')
        plt.plot(x_real, y_real, 'g-')
        plt.xlim(0.0, 1.0)
        plt.ylim(-2,2)
        plt.legend(('train', 'real', 'mean'), loc='upper left')
        title = "[PRML eq5] N:{0} / M:{1}".format(N, M)
        plt.title(title)
        plt.title("")
        plt.savefig('out/graph/PRML_p5.png')

    #plt.plot(Erms_ary)