import numpy as np

def dudt(u, bird):
    X = Fu(bird)/bird.m
    grav = -bird.g * np.sin(bird.theta)
    udot = X + grav
    udot += bird.r * bird.v - bird.q * bird.w

    return udot

def dvdt(v, bird):
    Y = Fv(bird)/bird.m
    grav = bird.g * np.cos(bird.theta) * np.sin(bird.phi)
    vdot = Y + grav
    vdot += bird.p * bird.w - bird.r * bird.u

    return vdot


def dwdt(w, bird):
    Z = Fw(bird)
    grav = bird.g * np.cos(bird.theta) * np.cos(bird.phi)
    wdot = Z + grav
    wdot += bird.q * bird.u - bird.p * bird.v

    return wdot

def dxdt(x, bird):
    theta = bird.theta
    phi = bird.phi
    psi = bird.psi
    return np.cos(theta) * np.cos(psi) * bird.u + (-np.cos(phi)*np.sin(psi) +\
            np.sin(phi) * np.sin(theta) * np.cos(psi)) * bird.v + \
            (np.sin(phi) * np.sin(psi) + np.cos(phi) * np.sin(theta) * \
            np.cos(psi)) * bird.w

def dydt(y, bird):
    theta = bird.theta
    phi = bird.phi
    psi = bird.psi
    return np.cos(theta) * np.sin(psi) * bird.u + (np.cos(phi) * np.cos(psi) + \
            np.sin(phi) * np.sin(theta) * np.sin(psi)) * bird.v + \
            (-np.sin(phi) * np.cos(psi) + np.cos(phi) * np.sin(theta) * \
            np.sin(psi)) * bird.w

def dzdt(z, bird):
    theta = bird.theta
    phi = bird.phi
    psi = bird.psi
    return (-np.sin(theta) * bird.u + np.sin(phi) * np.cos(theta) * \
            bird.v + np.cos(phi) * np.cos(theta) * bird.w)


def Fu(bird):
    #Drag = Cd * (rho * v^2)/2 * A
    D = bird.Cd * bird.Au * (bird.rho * bird.u ** 2)/2
    L = 0.0
    F = L - D
    return F

def Fv(bird):
    #Drag = Cd * (rho * v^2)/2 * A
    D = bird.Cd * bird.Av * (bird.rho * bird.v ** 2)/2
    L = 0.0
    F = L - D
    return F

def Fw(bird):
    #Lift = Cl * (rho * v^2)/2 * S
    #Drag = Cd * (rho * v^2)/2 * A
    D = bird.Cd * bird.Aw * (bird.rho * bird.w ** 2)/2
    L = bird.Cl * (bird.rho * bird.u**2)/2.0 * bird.S
    F = L - D
    return F
