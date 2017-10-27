
import numpy as np
import pylab

def phi(x):
        return np.array([x ** i for i in xrange(M+1)]).reshape((M+1,1))

def m(x, x_train, y_train, S):
    sum = np.array(np.zeros((M+1,1)))
    for n in xrange(len(x_train)):
        sum += phi(x_train[n])*y_train[n]

    return Beta * phi(x).T.dot(S).dot(sum)

def S(x_train):

    I = np.identity(M+1)
    Sigma = np.zeros((M+1,M+1))

    for n in xrange(len(x_train)):
        Sigma += np.dot(phi(x_train[n]), phi(x_train[n]).T)
        print 'Sigma'
        print Sigma.shape

    S_inv = alpha * I + Beta * Sigma
    S = np.linalg.inv(S_inv)
    return S

def s2(x, S):
    return 1/Beta + phi(x).T.dot(S).dot(phi(x))


def main(m_=6, beta_=11.1, alpha_=0.005, n_=80):
    global Beta, M, alpha, N
    Beta = beta_
    M = m_
    alpha = alpha_
    N = n_


    #Sine curve
    x_real = np.arange(0, 1, 0.01)
    y_real = np.sin(2*np.pi*x_real)

    ##Training Data
    x_train = np.linspace(0, 1, N)

    #Set "small Level of random noise having a Gaussian distribution"
    loc = 0
    scale = 0.3
    y_train = np.sin(2*np.pi*x_train) + np.random.normal(loc, scale, N)


    #Seeek predictive distribution corresponding to entire x
    mean = [m(x, x_train, y_train, S(x_train))[0,0] for x in x_real]
    variance = [s2(x, S(x_train))[0,0] for x in x_real]
    SD = np.sqrt(variance)
    upper = mean + SD
    lower = mean - SD

    pylab.plot(x_train, y_train, 'bo')
    pylab.plot(x_real, y_real, 'g-')
    pylab.plot(x_real, mean, 'r-')
    pylab.fill_between(x_real, upper, lower, color='pink')
    pylab.legend(('train','real','mean'), loc='upper left')
    pylab.xlim(0.0, 1.0)
    pylab.ylim(-2, 2)
    title ="[PRML eq1.69] N:{0} / M:{1} / Beta:{2:.02f} / Alpha:{3:.02f}".format(N, M, Beta, alpha)
    pylab.title(title)
    #pylab.show()
    pylab.savefig('out/graph/PRML_eq1_69.png')

if __name__ =='__main__':
    main()





