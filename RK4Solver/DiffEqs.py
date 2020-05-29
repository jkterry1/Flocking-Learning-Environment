import numpy as np

#sample diffeq used for testing purposes
def dydt_grav(t, y, agent, args):
    v0 = args['v0']
    a = args['a']
    return np.array([v0[j] + a[j] * t for j in range(0,3)])

#position
def dxdt(t, x, agent, args):
    return args['v']

#velocity
def dvdt(t, v, agent, args):
    F = args['F'](v)
    return F/args['m']

#angular momentum
def dLdt(t, L, agent, args):
    T = args['T'](L)
    return T - np.cross(args['w'], L)

#torque
def T(L):
    return np.array([0,0,0])

#force
def F(v):
    return np.array([0,0,0])
