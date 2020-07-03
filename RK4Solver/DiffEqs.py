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
    N = TN(bird, bird.r)
    L = TL(bird, p)
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
    M = TM(bird, q)
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
    N = TN(bird, r)
    L = TL(bird, bird.p)
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

def TL(bird, P):
    T = 0

    #torque due to upward lift on left wing
    #Lift = Cl * (rho * v^2)/2 * A
    if bird.u > 0:
        v = bird.u
        A = bird.Xl * bird.Yl * np.cos(bird.alpha_l)
        assert A >= 0
        r = bird.Xl/2.0
        Tl = np.cos(bird.beta_l) * r * bird.Cl * A * (bird.rho * v**2)/2.0
        assert Tl >= 0

        T += Tl

    #torque due to upward lift on right wing
    if bird.u > 0:
        v = bird.u
        A = bird.Xl * bird.Yl * np.cos(bird.alpha_r)
        assert A > 0
        r = bird.Xl/2.0
        Tr = -np.cos(bird.beta_r) * r * bird.Cl * A * (bird.rho * v**2)/2.0
        assert Tr <= 0

        T += Tr

    #torque due to lift in v direction from left wing
    if bird.u > 0:
        v = bird.u
        A = bird.Xl * bird.Yl * np.cos(bird.alpha_l)
        assert A > 0
        r = abs(np.sin(bird.beta_l)) * bird.Xl/2.0
        Tl = abs(np.sin(bird.beta_l)) * r * bird.Cl * A * (bird.rho * v**2)/2.0
        assert Tl >= 0

        T += Tl

    #torque due to lift in v direction from right wing
    if bird.u > 0:
        v = bird.u
        A = bird.Xl * bird.Yl * np.cos(bird.alpha_r)
        assert A > 0
        r = abs(np.sin(bird.beta_r)) * bird.Xl/2.0
        Tr = -abs(np.sin(bird.beta_r)) * r * bird.Cl * A * (bird.rho * v**2)/2.0
        assert Tr <= 0
        assert np.sign(Tl) == 0 or np.sign(Tl) != np.sign(Tr)

        T += Tr

    #drag torque due to rotation on left wing
    #Td = cd * A * (rho * v^2)/2 * r
    #v = wr
    v = P * bird.Xl/2.0
    r = bird.Xl/2.0
    A = bird.Xl * bird.Yl
    assert A >= 0
    Tl = -np.sign(bird.p) * r * A * bird.Cd * (bird.rho * v**2)/2.0
    assert np.sign(Tl) == 0 or np.sign(Tl) != np.sign(bird.p)

    T += Tl

    #drag due to rotation on right wing
    v = P * bird.Xl/2.0
    r = bird.Xl/2.0
    A = bird.Xl * bird.Yl
    assert A >= 0
    Tr = -np.sign(bird.p) * r * A * bird.Cd * (bird.rho * v**2)/2.0
    assert np.sign(Tr) == 0 or np.sign(Tr) != np.sign(bird.p)

    T += Tr

    #torque due to vortices
    T += bird.vortex_torque_u

    bird.T[0] = T
    return T

def TM(bird, Q):
    T = 0
    r = bird.Yl/4.0
    v = (Q * r)/2.0
    A = (bird.Xl * bird.Yl)
    #rotation drag
    D = -np.sign(v) * r * bird.Cd * A * (bird.rho * v**2)/2.0
    T += 2.0 * D
    T += bird.vortex_torque_v
    bird.T[1] = T
    return T

def TN(bird, R):
    T = 0

    #drag from left wing (including alpha rotation)
    #Td = cd * A * (rho * v^2)/2 * r
    v = bird.u
    r = bird.Xl/2.0
    A = bird.Xl * abs(np.sin(bird.alpha_l)) * bird.Yl
    assert A >= 0
    Tdl = np.sign(bird.u) * r * bird.Cd * A * (bird.rho * bird.u**2)/2.0

    T += Tdl

    #drag on right wing (including alpha rotation)
    #Td = cd * A * (rho * v^2)/2 * r
    v = bird.u
    r = bird.Xl/2.0
    A = bird.Xl * abs(np.sin(bird.alpha_r)) * bird.Yl
    assert A >= 0
    Tdr = -np.sign(bird.u) * r * bird.Cd * A * (bird.rho * v**2)/2.0

    T += Tdr

    #drag from rotation on left wing
    #Td = cd * A * (rho * v^2)/2 * r
    #v = wr
    v = R * bird.Xl/2.0
    r = bird.Xl/2.0
    A = bird.Xl * bird.Yl * abs(np.sin(bird.alpha_l)) + bird.Yl * bird.Zl * np.cos(bird.alpha_r)
    assert A >= 0
    Tdl = -np.sign(bird.r) * r * bird.Cd * (bird.rho * v**2)/2.0
    assert np.sign(Tdl) == 0 or np.sign(Tdl) != np.sign(bird.r)

    T += Tdl

    #drag from rotation on right wing
    #Td = cd * A * (rho * v^2)/2 * r
    #v = wr
    v = R * bird.Xl/2.0
    r = bird.Xl/2.0
    A = bird.Xl * bird.Yl * abs(np.sin(bird.alpha_r)) + bird.Yl * bird.Zl * np.cos(bird.alpha_r)
    assert A >= 0
    Tdr = -np.sign(bird.r) * r * bird.Cd * (bird.rho * v**2)/2.0
    assert np.sign(Tdr) == 0 or np.sign(Tdl) != np.sign(bird.r)

    T += Tdr

    #drag from vortices
    T += bird.vortex_torque_w

    bird.T[2] = T
    return T

def Fu(u, bird):
    F = 0
    alpha_l = abs(bird.alpha_l)
    alpha_r = abs(bird.alpha_r)

    #Drag = Cd * (rho * v^2)/2 * A
    A = bird.Xl * bird.Zl + bird.Xl * bird.Yl * np.sin(alpha_l) +\
        bird.Xl * bird.Yl * np.sin(alpha_r)
    D = -np.sign(u) * bird.Cd * A * (bird.rho * u**2)/2.0

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
