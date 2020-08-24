import numpy as np

def drdt(gamma, epsilon, b, theta):
    return (gamma/(4.0* np.pi * epsilon)) * b * np.tan(theta/2.0)

def duvwdt(uvw, bird):
    theta = bird.theta
    phi = bird.phi
    psi = bird.psi
    pqr = bird.pqr
    u, v, w = uvw
    m = bird.m
    g = bird.g
    grav = np.array([-g * np.sin(bird.theta),
                    g * np.cos(bird.theta) * np.sin(bird.phi),
                    g * np.cos(bird.theta) * np.cos(bird.phi)])

    #print("grav force: ", grav)

    F = np.array([Fu(u, bird), Fv(v, bird), Fw(w, bird)])
    return (1.0/m)*F - np.cross(pqr, uvw) + grav

def dxyzdt(xyz, bird):
    theta = bird.theta
    phi = bird.phi
    psi = bird.psi
    R3 = np.array([[np.cos(psi), -np.sin(psi), 0.0],
                    [np.sin(psi), np.cos(psi), 0.0],
                    [0.0, 0.0, 1.0]])
    R2 = np.array([[np.cos(theta), 0.0, np.sin(theta)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(theta), 0.0, np.cos(theta)]])
    R1 = np.array([[1.0, 0.0, 0.0],
                    [0.0, np.cos(phi), -np.sin(phi)],
                    [0.0, np.sin(phi), np.cos(phi)]])
    output = np.matmul(R3, np.matmul(R2, np.matmul(R1, bird.uvw)))
    return output


def dpqrdt(pqr, bird):
    p, q, r = pqr
    LMN = np.array([TL(bird, p), TM(bird, q), TN(bird, r)])
    inertia = bird.inertia
    mat = np.array([[0.0, -r, q],
                    [r, 0.0, -p],
                    [-q, p, 0.0]])
    rhs = LMN - np.matmul(mat, np.matmul(inertia, pqr))
    return np.matmul(np.linalg.inv(inertia), rhs)


def danglesdt(angles, bird):
    phi, theta, psi = angles
    pqr = bird.pqr
    mat = np.array([[1.0, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
                    [0.0, np.cos(phi), -np.sin(phi)],
                    [0.0, np.sin(phi)*(1.0/np.cos(theta)), np.cos(phi)*(1.0/np.cos(theta))]])
    output = np.matmul(mat, pqr)
    return output

def TL(bird, P):
    T = 0

    #torque due to upward lift on left wing
    #Lift = Cl * (rho * v^2)/2 * A
    if bird.u > 0:
        v = bird.u
        A = bird.Xl * bird.Yl * np.cos(bird.alpha_l)
        cl, cr = C_lift(bird)
        check_A(bird, A, bird.alpha_r)
        r = bird.Xl/2.0
        Tl = np.cos(bird.beta_l) * r * cl * A * (bird.rho * v**2)/2.0
        #assert Tl >= 0

        T += Tl

    #torque due to upward lift on right wing
    if bird.u > 0:
        v = bird.u
        A = bird.Xl * bird.Yl * np.cos(bird.alpha_r)
        cl, cr = C_lift(bird)
        check_A(bird, A, bird.alpha_r)
        r = bird.Xl/2.0
        Tr = -np.cos(bird.beta_r) * r * cr * A * (bird.rho * v**2)/2.0
        #assert Tr <= 0

        T += Tr

    #torque due to lift in v direction from left wing
    if bird.u > 0:
        v = bird.u
        A = bird.Xl * bird.Yl * np.cos(bird.alpha_l)
        cl, cr = C_lift(bird)
        check_A(bird, A, bird.alpha_l)
        r = abs(np.sin(bird.beta_l)) * bird.Xl/2.0
        Tl = abs(np.sin(bird.beta_l)) * r * cl * A * (bird.rho * v**2)/2.0
        #assert Tl >= 0

        T += Tl

    #torque due to lift in v direction from right wing
    if bird.u > 0:
        v = bird.u
        A = bird.Xl * bird.Yl * np.cos(bird.alpha_r)
        cl, cr = C_lift(bird)
        check_A(bird, A, bird.alpha_r)
        r = abs(np.sin(bird.beta_r)) * bird.Xl/2.0
        Tr = -abs(np.sin(bird.beta_r)) * r * cr * A * (bird.rho * v**2)/2.0
        #assert Tr <= 0
        #assert np.sign(Tl) == 0 or np.sign(Tl) != np.sign(Tr)

        T += Tr

    #drag torque due to rotation on left wing
    #Td = cd * A * (rho * v^2)/2 * r
    #v = wr
    v = P * bird.Xl/2.0
    r = bird.Xl/2.0
    A = bird.Xl * bird.Yl
    check_A(bird, A, 0.0)
    cl, cr = C_lift(bird)
    Tl = -np.sign(bird.p) * r * A * cl * (bird.rho * v**2)/2.0
    #assert np.sign(Tl) == 0 or np.sign(Tl) != np.sign(bird.p)

    T += Tl

    #drag due to rotation on right wing
    v = P * bird.Xl/2.0
    r = bird.Xl/2.0
    A = bird.Xl * bird.Yl
    cl, cr = C_lift(bird)
    #assert A >= 0
    Tr = -np.sign(bird.p) * r * A * cr* (bird.rho * v**2)/2.0
    #assert np.sign(Tr) == 0 or np.sign(Tr) != np.sign(bird.p)

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
    check_A(bird, A, bird.alpha_l)
    Tdl = np.sign(bird.u) * r * bird.Cd * A * (bird.rho * bird.u**2)/2.0

    T += Tdl

    #drag on right wing (including alpha rotation)
    #Td = cd * A * (rho * v^2)/2 * r
    v = bird.u
    r = bird.Xl/2.0
    A = bird.Xl * abs(np.sin(bird.alpha_r)) * bird.Yl
    check_A(bird, A, bird.alpha_r)
    Tdr = -np.sign(bird.u) * r * bird.Cd * A * (bird.rho * v**2)/2.0

    T += Tdr

    #drag from rotation on left wing
    #Td = cd * A * (rho * v^2)/2 * r
    #v = wr
    v = R * bird.Xl/2.0
    r = bird.Xl/2.0
    A = bird.Xl * bird.Yl * abs(np.sin(bird.alpha_l)) + bird.Yl * bird.Zl * np.cos(bird.alpha_r)
    check_A(bird, A, bird.alpha_r)
    Tdl = -np.sign(bird.r) * r * bird.Cd * (bird.rho * v**2)/2.0
    #assert np.sign(Tdl) == 0 or np.sign(Tdl) != np.sign(bird.r)

    T += Tdl

    #drag from rotation on right wing
    #Td = cd * A * (rho * v^2)/2 * r
    #v = wr
    v = R * bird.Xl/2.0
    r = bird.Xl/2.0
    A = bird.Xl * bird.Yl * abs(np.sin(bird.alpha_r)) + bird.Yl * bird.Zl * np.cos(bird.alpha_r)
    check_A(bird, A, bird.alpha_r)
    Tdr = -np.sign(bird.r) * r * bird.Cd * (bird.rho * v**2)/2.0
    #assert np.sign(Tdr) == 0 or np.sign(Tdl) != np.sign(bird.r)

    T += Tdr

    #drag from vortices
    T += bird.vortex_torque_w

    bird.T[2] = T
    return T

def Fu(u, bird):
    F = 0.0
    alpha_l = abs(bird.alpha_l)
    alpha_r = abs(bird.alpha_r)

    #Drag = Cd * (rho * v^2)/2 * A
    A = bird.Xl * bird.Zl + bird.Xl * bird.Yl * np.sin(alpha_l) +\
        bird.Xl * bird.Yl * np.sin(alpha_r)
    #print("u before drag: ", u)
    D = -np.sign(u) * bird.Cd * A * (bird.rho * u**2)/2.0
    #print("Drag: ", D)

    F += D
    F += bird.vortex_force_u
    #print("Vortex force: ", bird.vortex_force_u)
    F += bird.thrust
    #print("Thrust: ", bird.thrust)
    bird.F[0] = F
    return F

def Fv(v, bird):
    F = 0
    #lift
    A = bird.Xl * bird.Yl
    cl, cr = C_lift(bird)
    L_left = cl * A * (bird.rho * bird.u**2)/2.0
    L_right = cr * A * (bird.rho * bird.u**2)/2.0
    #Drag = Cd * (rho * v^2)/2 * A
    A = bird.Yl * bird.Zl
    D = -np.sign(v) * bird.Cd * A * (bird.rho * v ** 2)/2.0
    if bird.u > 0.0:
        F += L_left * np.sin(bird.beta_l)
        F += -L_right * np.sin(bird.beta_r)
    F += D
    F += bird.vortex_force_v
    bird.F[1] = F
    return F

def Fw(w, bird):
    F = 0
    #Lift = Cl * (rho * v^2)/2 * S
    #Drag = Cd * (rho * v^2)/2 * A
    A = bird.Xl * bird.Yl
    cl, cr = C_lift(bird)
    L_left = cl * A * (bird.rho * bird.u**2)/2.0
    L_right = cr * A * (bird.rho * bird.u**2)/2.0
    D = -np.sign(w) * bird.Cd * A * (bird.rho * w ** 2)/2.0
    if bird.u > 0.0:
        #left wing lift
        F += L_left * np.cos(bird.beta_l)
        #right wing lift
        F += L_right * np.cos(bird.beta_r)
    F += D
    F += bird.vortex_force_w
    bird.F[2] = F
    return F

def check_A(bird, A, angle):
    if A < 0:
        bird.print_bird(bird)
        file = open('A_errors.txt', 'w')
        file = sys.stderr
        #print("Angle: ", angle, file = file)
        bird.broken = True

def C_lift(bird):
    if bird.u == 0:
        d = np.pi/2.0
    else:
        d = np.arctan(bird.w/bird.u)
    aoa_l = np.degrees(bird.alpha_l + d)
    aoa_r = np.degrees(bird.alpha_r + d)
    c_max = bird.Cl_max
    if aoa_l < -10 or aoa_l > 25:
        cl = 0
    elif aoa_l < 15:
        cl = ((c_max/25.0) * aoa_l) + (10.0 * c_max / 25.0)
    elif aoa_l < 20:
        cl = c_max
    else:
        cl = ((-c_max/25.0) * aoa_l) + (c_max + (20.0 * c_max / 25.0))

    if aoa_r < -10 or aoa_r > 25:
        cr = 0
    elif aoa_r < 15:
        cr = ((c_max/25.0) * aoa_r) + (10.0 * c_max / 25.0)
    elif aoa_r < 20:
        cr = c_max
    else:
        cr = ((-c_max/25.0) * aoa_r) + (c_max + (20.0 * c_max / 25.0))

    return cl, cr
