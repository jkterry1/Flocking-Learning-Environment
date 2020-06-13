import numpy as np

def dudt(u, bird):
    X = Fu(u, bird)/bird.m
    grav = -bird.g * np.sin(bird.theta)
    udot = X + grav
    udot += bird.r * bird.v - bird.q * bird.w

    return udot

def dvdt(v, bird):
    Y = Fv(v, bird)/bird.m
    grav = bird.g * np.cos(bird.theta) * np.sin(bird.phi)
    vdot = Y + grav
    vdot += bird.p * bird.w - bird.r * bird.u

    return vdot


def dwdt(w, bird):
    Z = Fw(w, bird)
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

def dpdt(p, bird):
    # 1/Ixx * (L + (Iyy - Izz) * q * r)  # = pdot
    Ix = bird.Ixx
    Iy = bird.Iyy
    Iz = bird.Izz
    Ixz = bird.Ixz
    q = bird.q
    r = bird.r
    N = TN(bird)
    L = TL(bird)
    pdot = 1.0/(Ix - (Ixz/Iz)**2)
    pdot *= (L + q * r * (Iy - Iz) - Ixz * p * q + \
            (Ixz/Iz) * (N - p * q * (Iy - Ix) - Ixz * q * r) )
    return pdot

def dqdt(q, bird):
    # 1/Iyy * (M + (Izz - Ixx) * p * r)  # = qdot
    Ix = bird.Ixx
    Iy = bird.Iyy
    Iz = bird.Izz
    Ixz = bird.Ixz
    p = bird.p
    r = bird.r
    M = TM(bird)
    qdot = (1.0/Iy)
    qdot*= (M - r * p * (Ix - Iz) - Ixz * (p**2 - r**2))
    return qdot


def drdt(r, bird):
    # 1/Izz * (N + (Ixx - Iyy) * p * q)  # = rdot
    Ix = bird.Ixx
    Iy = bird.Iyy
    Iz = bird.Izz
    Ixz = bird.Ixz
    p = bird.p
    q = bird.q
    N = TN(bird)
    L = TL(bird)
    rdot = 1.0/(Iz - (Ixz**2 / Ix))
    rdot *= (N + (Ixz / Ix) * (q * r * (Iz - Iy) - Ixz * p * q + L) -\
            p * q * (Iy - Ix) - Ixz * q * r)
    return rdot

def dthetadt(theta, bird):
    return bird.q * np.cos(bird.phi) - bird.r * np.sin(bird.phi)

def dphidt(phi, bird):
    return bird.p + bird.q * np.sin(bird.phi) * np.tan(bird.theta) + \
            bird.r * np.cos(bird.phi) * np.tan(bird.theta)

def dpsidt(psi, bird):
    return (bird.q * np.sin(bird.phi) + \
            bird.r * np.cos(bird.phi)) * (1.0/np.cos(bird.theta))

def TL(bird):
    # Drag = Cd * (rho * v^2)/2 * A
    '''
    v = (bird.p * bird.r)/2.0
    A = (bird.Xl * bird.Yl)
    r = bird.Xl/2.0
    T = -r * bird.Cd * A * (bird.rho * v**2)/2.0
    return T
    '''
    return 0.0


def TM(bird):
    return 0.0

def TN(bird):
    return 0.0

def Fu(u, bird):
    #Drag = Cd * (rho * v^2)/2 * A
    A = bird.Xl * bird.Yl
    D = bird.Cd * A * (bird.rho * u**2)/2.0
    F = -D
    return F

def Fv(v, bird):
    #Drag = Cd * (rho * v^2)/2 * A
    D = bird.Cd * bird.Av * (bird.rho * v ** 2)/2.0
    F = -D
    return F

def Fw(w, bird):
    #Lift = Cl * (rho * v^2)/2 * S
    #Drag = Cd * (rho * v^2)/2 * A
    D = bird.Cd * bird.Aw * (bird.rho * w ** 2)/2.0
    L = bird.Cl * (bird.rho * bird.u**2)/2.0 * bird.S
    F = L - D
    return F
