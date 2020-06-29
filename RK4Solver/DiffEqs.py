import numpy as np

def dudt(u, bird):
    X = Fu(u, bird)/bird.m
    grav = -bird.g * np.sin(bird.theta)
    udot = X + grav
    udot += bird.r * bird.v - bird.q * bird.w

    return udot

def dvdt(v, bird):
    #g * sphi*cthe - r * ub + p * wb # = vdot
    Y = Fv(v, bird)/bird.m
    grav = bird.g * np.cos(bird.theta) * np.sin(bird.phi)
    vdot = Y + grav
    vdot += bird.p * bird.w - bird.r * bird.u

    return vdot


def dwdt(w, bird):
    Z = Fw(w, bird)/bird.m
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
    #cthe*spsi * ub + (cphi*cpsi+sphi*sthe*spsi) * vb + (-sphi*cpsi+cphi*sthe*spsi) * wb # = yEdot
    theta = bird.theta
    phi = bird.phi
    psi = bird.psi
    return np.cos(theta) * np.sin(psi) * bird.u + \
            (np.cos(phi) * np.cos(psi) + np.sin(phi) * np.sin(theta) * np.sin(psi)) * bird.v + \
            (-np.sin(phi) * np.cos(psi) + np.cos(phi) * np.sin(theta) * np.sin(psi)) * bird.w

def dzdt(z, bird):
    theta = bird.theta
    phi = bird.phi
    psi = bird.psi
    return (-np.sin(theta) * bird.u + np.sin(phi) * np.cos(theta) * \
            bird.v + np.cos(phi) * np.cos(theta) * bird.w)

def dpdt(p, bird):
    #1/Ixx * (L + (Iyy - Izz) * q * r)  # = pdot
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
    #1/Iyy * (M + (Izz - Ixx) * p * r)  # = qdot
    Ix = bird.Ixx
    Iy = bird.Iyy
    Iz = bird.Izz
    Ixz = bird.Ixz
    p = bird.p
    r = bird.r
    M = TM(bird)
    print("M ", M)
    print("val ", r * p * (Iz - Ix))
    qdot = (1.0/Iy)
    qdot *= (M + r * p * (Iz - Ix) - Ixz * (p**2 - r**2))
    return qdot


def drdt(r, bird):
    #xdot[5] = 1/Izz * (N + (Ixx - Iyy) * p * q)  # = rdot
    Ix = bird.Ixx
    Iy = bird.Iyy
    Iz = bird.Izz
    Ixz = bird.Ixz
    p = bird.p
    q = bird.q
    N = TN(bird)
    L = TL(bird)
    rdot = 1.0/(Iz - (Ixz**2 / Ix))
    rdot *= (N + p * q * (Ix - Iy) + (Ixz / Ix) *\
            (q * r * (Iz - Iy) - Ixz * p * q + L) - Ixz * q * r)
    return rdot

def dthetadt(theta, bird):
    #q * cphi - r * sphi  # = thetadot
    return bird.q * np.cos(bird.phi) - bird.r * np.sin(bird.phi)

def dphidt(phi, bird):
    #p + (q*sphi + r*cphi) * sthe / cthe  # = phidot
    return bird.p + (bird.q * np.sin(bird.phi) + \
            bird.r * np.cos(bird.phi)) * (np.sin(bird.theta)/np.cos(bird.theta))

def dpsidt(psi, bird):
    #(q * sphi + r * cphi) / cthe  # = psidot
    return (bird.q * np.sin(bird.phi) + \
            bird.r * np.cos(bird.phi)) * (1.0/np.cos(bird.theta))

def TL(bird):
    T = 0
    r = bird.Xl/2.0
    #lift
    A = bird.Xl * bird.Yl
    L = bird.Cl * A * (bird.rho * bird.u**2)/2.0
    if bird.u > 0.0:
        #left wing lift
        Tl = L * np.cos(bird.alpha_l) * np.cos(bird.beta_l) * r
        #right wing lift
        Tr = -L * np.cos(bird.alpha_r) * np.cos(bird.beta_r) * r

        #left wing lift, v component, times r
        Tl += L * np.sin(bird.beta_l) * bird.Xl * np.sin(bird.beta_l)
        #right wing lift, v component
        Tr += -L * np.sin(bird.beta_r) * bird.Xl * np.sin(bird.beta_r)

    # v = rw
    # T = F * L
    v = (bird.p * r)/2.0
    A = (bird.Xl * bird.Yl)

    #drag torque from rotation
    D = -np.sign(v) * r * bird.Cd * A * (bird.rho * v**2)/2.0
    T += 2.0 * D
    T += bird.vortex_torque_u
    bird.T[0] = T
    return T

def TM(bird):
    T = 0
    r = bird.Yl/4.0
    v = (bird.q * r)/2.0
    A = (bird.Xl * bird.Yl)
    #rotation drag
    D = -np.sign(v) * r * bird.Cd * A * (bird.rho * v**2)/2.0
    T += 2.0 * D
    T += bird.vortex_torque_v
    bird.T[1] = T
    return T

def TN(bird):
    T = 0
    len = bird.Xl/2.0
    alpha_l = abs(bird.alpha_l)
    alpha_r = abs(bird.alpha_r)
    Fl = 0.0
    Fr = 0.0

    #drag contribution from u direction motion
    A = bird.Yl * bird.Xl * np.sin(alpha_l) + bird.Xl * bird.Zl
    Fl += np.sign(bird.u) * bird.Cd * A * (bird.rho * bird.u**2)/2.0

    #drag contribution from u direction motion
    A = bird.Yl * bird.Xl * np.sin(alpha_r) + bird.Xl * bird.Zl
    Fr += -np.sign(bird.u) * bird.Cd * A * (bird.rho * bird.u**2)/2.0

    #add wing orientation contributions to the torque
    T += Fl * len
    T += Fr * len

    #drag that slows rotation
    v = (bird.r * len)/2.0
    A_l = (bird.Yl * bird.Zl) + bird.Xl * bird.Yl * np.sin(alpha_l)
    D = -np.sign(v) * bird.Cd * A_l * (bird.rho * v**2)/2.0
    A_r = (bird.Yl * bird.Zl) + bird.Xl * bird.Yl * np.sin(alpha_r)
    D += -np.sign(v) * bird.Cd * A_l * (bird.rho * v**2)/2.0
    T += D * len

    #drag from vortices
    T += bird.vortex_torque_w

    bird.T[2] = T
    return T

def Fu(u, bird):
    F = 0
    alpha_l = abs(bird.alpha_l)
    alpha_r = abs(bird.alpha_r)
    #lift
    # A = bird.Xl * bird.Yl
    # L = bird.Cl * A * (bird.rho * u**2)/2.0
    #Drag = Cd * (rho * v^2)/2 * A
    A = bird.Xl * bird.Zl + bird.Xl * bird.Yl * np.sin(alpha_l) + bird.Xl * bird.Yl * np.sin(alpha_r)
    D = -np.sign(u) * bird.Cd * A * (bird.rho * u**2)/2.0
    # if bird.u > 0.0:
    #     F += L * np.sin(bird.alpha_l)
    #     F += L * np.sin(bird.alpha_r)
    #print("Forward force from alpha ", L * np.sin(bird.alpha_r))
    F += D
    F += bird.vortex_force_u
    F += bird.thrust
    bird.F[0] = F
    return F

def Fv(v, bird):
    F = 0
    #lift
    A = bird.Xl * bird.Yl
    L = bird.Cl * A * (bird.rho * bird.u**2)/2.0
    #Drag = Cd * (rho * v^2)/2 * A
    A = bird.Yl * bird.Zl
    D = -np.sign(v) * bird.Cd * A * (bird.rho * v ** 2)/2.0
    if bird.u > 0.0:
        F += L * np.sin(bird.beta_l)
        F += -L * np.sin(bird.beta_r)
    F += D
    F += bird.vortex_force_v
    bird.F[1] = F
    return F

def Fw(w, bird):
    F = 0
    #Lift = Cl * (rho * v^2)/2 * S
    #Drag = Cd * (rho * v^2)/2 * A
    A = bird.Xl * bird.Yl
    L = bird.Cl * A * (bird.rho * bird.u**2)/2.0
    D = -np.sign(w) * bird.Cd * A * (bird.rho * w ** 2)/2.0
    if bird.u > 0.0:
        #left wing lift
        F += L * np.cos(bird.beta_l)
        #right wing lift
        F += L * np.cos(bird.beta_r)
    F += D
    F += bird.vortex_force_w
    bird.F[2] = F
    return F
