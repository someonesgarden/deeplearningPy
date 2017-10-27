#!/usr/bin/env python3

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD


print("keras_version:")
print(keras.__version__)


# model = Sequential([
#     Dense(input_dim=2, units=1),
#     Activation('sigmoid')
# ])

model = Sequential()
model.add(Dense(input_dim=2, units=1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))

X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[1]])

model.fit(X, Y, epochs=200, batch_size=1)
classes = model.predict_classes(X, batch_size=1)
prob = model.predict_proba(X, batch_size=1)

print("classified:")
print(Y==classes)
print()
print("output probability:")
print(prob)
