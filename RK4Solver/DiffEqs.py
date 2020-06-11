import numpy as np

#sample diffeq used for testing purposes
def dudt(u, t, bird):
    X = Fx()
    grav = bird.m * bird.g * np.sin(bird.theta)
    udot = X - grav
    udot += bird.m * bird.r * bird.v - bird.m * bird.q * bird.w
    udot = udot/bird.m

    return udot

def dvdt(v, t, bird):
    Y = Fy()
    grav = bird.m * bird.g * np.cos(bird.theta) * np.sin(bird.phi)
    vdot = Y + grav
    vdot += bird.m * bird.p * bird.w - bird.m * bird.r * bird.u
    vdot = vdot/bird.m

    return vdot


def dwdt(w, t, bird):
    Z = Fz()
    grav = bird.m * bird.g * np.cos(bird.theta) * np.cos(bird.phi)
    wdot = Z + grav
    wdot += bird.m * bird.q * bird.u - bird.m * bird.p * bird.v
    wdot = wdot/bird.m

    return wdot

def dxdt(vx, t, bird):
    return vx

def dydt(vy, t, bird):
    return vy

def dzdt(vz, t, bird):
    return vz


def Fx():
    return 0.0

def Fy():
    return 0.0

def Fz():
    return 0.0
