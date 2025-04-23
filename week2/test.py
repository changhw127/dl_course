# -*- coding: utf-8 -*-
import numpy as np


def relu(z):
    z[z < 0] = 0
    return z





def neuron(w, x):
    return w.T @ x


x = np.array([1,  0.5,  -1])
w = np.array([0.5, 0.25, -0.75])
print(neuron(w, x))
# 1.375
