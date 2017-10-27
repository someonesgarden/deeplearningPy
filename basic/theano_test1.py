#!/usr/bin/env python3
import numpy as np
import theano
import theano.tensor as T

c = T.scalar('c')
v = T.vector('v')
A = T.matrix('A')
w = A.dot(v)

matrix_times_vector = theano.function(inputs=[A, v], outputs=w)

A_val = np.array([[1,2],[3,4]])
v_val = np.array([5,6])
w_val = matrix_times_vector(A_val,v_val)

print(w_val)

x = theano.shared(-10000.0, 'x')

cost = x*x + x + 1
x_update = x - 0.3*T.grad(cost, x)


train = theano.function(inputs=[], outputs=cost, updates=[(x, x_update)])

for i in range(25):
    cost_val = train()
    print(cost_val)


print(x.get_value())