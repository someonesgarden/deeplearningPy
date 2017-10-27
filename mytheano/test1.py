import theano.tensor as T

x = T.iscalar('X')
x = T.scalar('X', dtype='int32')
v = T.fvector('v')
m = T.dmatrix('m')
t = T.dtensor3('t')

y = 2*x
