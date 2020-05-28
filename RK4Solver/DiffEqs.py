import numpy as np

def dydt_grav(t, y, i, args):
    v0, a = args
    v0 = v0[i]
    a = a[i]
    return np.array([v0[j] + a[j] * t for j in range(0,3)])
