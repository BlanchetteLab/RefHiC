import cooler
import numpy as np
import numba
from numba import jit
from numba.experimental import jitclass


# @jit()
def relative_right_shift(x):
    b, h, w = x.shape
    output = np.zeros((b,h,2*w))
    output[:b,:h,:w] = x
    output=output.reshape(b,-1)[:,:-h].reshape(b,h,-1)[:,:,h-1:]
    return output
x = np.zeros((2,21,21))
for i in range(10000000):
    relative_right_shift(x)