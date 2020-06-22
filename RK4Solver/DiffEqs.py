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
    return np.cos(theta) * np.sin(psi) * bird.v + \
            (np.cos(phi) * np.cos(psi) + np.sin(phi) * np.sin(theta) * np.sin(psi)) * bird.v + \
            (-np.sin(phi) * np.cos(psi) + np.cos(phi) * np.sin(theta) * \
            np.sin(psi)) * bird.w

def dzdt(z, bird):
    theta = bird.theta
    phi = bird.phi
    psi = bird.psi
    return (-np.sin(theta) * bird.u + np.sin(phi) * np.cos(theta) * \
            bird.v + np.cos(phi) * np.cos(theta) * bird.w)

def dpdt(p, bird):
    Ix = bird.Ixx
    Iy = bird.Iyy
    Iz = bird.Izz
    Ixz = bird.Ixz
    q = bird.q
    r = bird.r
    N = TN(bird)
    L = TL(bird)
    pdot = 1.0/(Ix - (Ixz/Iz)**2)
    pdot *= (L + (q * r * (Iy - Iz)) - Ixz * p * q + \
            (Ixz/Iz) * (N - p * q * (Iy - Ix) - Ixz * q * r) )
    return pdot

def dqdt(q, bird):
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
    return bird.p + (bird.q * np.sin(bird.phi) + \
            bird.r * np.cos(bird.phi)) * (np.sin(bird.theta)/np.cos(bird.theta))

def dpsidt(psi, bird):
    return (bird.q * np.sin(bird.phi) + \
            bird.r * np.cos(bird.phi)) * (1.0/np.cos(bird.theta))

def TL(bird):
    # v = rw
    # T = F * L
    r = bird.Xl/2.0
    v = (bird.p * r)/2.0
    A = (bird.Xl * bird.Yl)/2.0
    T = -np.sign(v) * r * bird.Cd * A * (bird.rho * v**2)/2.0
    T = 2.0 * T + bird.Tp + bird.vortex_torque_u
    bird.T[0] = T
    return T

def TM(bird):
    r = bird.Yl/2.0
    v = (bird.q * r)/2.0
    A = (bird.Xl * bird.Yl)/2.0
    T = -np.sign(v) * r * bird.Cd * A * (bird.rho * v**2)/2.0
    T = 2.0 * T + bird.Tq + bird.vortex_torque_v
    bird.T[1] = T
    return T

def TN(bird):
    r = bird.Xl/2.0
    v = (bird.r * r)/2.0
    A = (bird.Xl * bird.Zl)/2.0
    T = -np.sign(v) * r * bird.Cd * A * (bird.rho * v**2)/2.0
    T = 2.0 * T + bird.Tr + bird.vortex_torque_w
    bird.T[2] = T
    return T

def Fu(u, bird):
    #Drag = Cd * (rho * v^2)/2 * A
    A = bird.Xl * bird.Zl
    D = np.sign(u) * bird.Cd * A * (bird.rho * u**2)/2.0
    F = -D + bird.vortex_force_u
    bird.F[0] = F
    return F + bird.thrust

def Fv(v, bird):
    #Drag = Cd * (rho * v^2)/2 * A
    A = bird.Yl * bird.Zl
    D = np.sign(v) * bird.Cd * A * (bird.rho * v ** 2)/2.0
    F = -D + bird.vortex_force_v
    bird.F[1] = F
    return F

def Fw(w, bird):
    #Lift = Cl * (rho * v^2)/2 * S
    #Drag = Cd * (rho * v^2)/2 * A
    A = bird.Xl * bird.Yl
    D = np.sign(w) * bird.Cd * A * (bird.rho * w ** 2)/2.0
    L = bird.Cl * A * (bird.rho * bird.u**2)/2.0
    F = L - D + bird.vortex_force_w
    bird.F[2] = F
    return F
