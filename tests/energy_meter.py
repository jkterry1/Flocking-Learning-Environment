import numpy as np
import matplotlib.pyplot as plt

'''
The energy function will calculate the energy to travel one meter in the forward direction
for any velocity (V given as a 3-vector of velocity) and default wing positions and orientations.
This doesn't account for thrust, vortices, or orientation and is just an estimate.
'''

m = 5.0        #goose is ~5kg
Cl_max = 1.6   #experimental value
Cd = 0.3       #experimental value

Xl = 0.80     #approximate dimensions
Yl = 0.35
Zl = 0.8
rho = 1.225
alpha_l = 0.0
alpha_r = 0.0
beta_l = 0.0
beta_r = 0.0

def energy(V):
    F = [0.0, 0.0, 0.0]
    u, v, w = V
    F[0] = Fu(u)
    F[1] = Fv(u,v)
    F[2] = Fw(u,w)

    #time to travel one meter
    t = 1.0/u

    return abs(F[0] * u * t) + abs(F[1] * v * t) + abs(F[2] * w * t)



def Fu(u):
    #Calculate F[0] (forward direction)
    #Area of the wing that is perpendicular to the forwad direction of motion.
    A = Xl * Zl + Xl * Yl * np.sin(alpha_l) + Xl * Yl * np.sin(alpha_r)
    D = -np.sign(u) * Cd * A * (rho * u**2/2.0)
    return D

def Fv(u,v):
    #calculate F[1] (sideways direction)
    F = 0.0
    A = Xl * Yl
    cl = 10.0 * Cl_max / 25.0
    cr = 10.0 * Cl_max / 25.0

    #Lift on the left and right wongs
    L_left = cl * A * (rho * v**2)/2.0
    L_right = cr * A * (rho * v**2)/2.0

    A = Yl * Zl;
    #Drag on the bird
    D = -np.sign(v) * Cd * A * (rho * v**2/2.0);

    #If the bird is moving forward, there is a lift contribution to the force
    #from each wing.
    if (u > 0.0):
        F += L_left * np.sin(beta_l);
        F += -L_right * np.sin(beta_r);

    F += D
    return F

def Fw(u,w):
    F = 0.0
    A = Xl * Yl;
    cl = 10.0 * Cl_max / 25.0
    cr = 10.0 * Cl_max / 25.0

    #Lift due to left and right wing
    L_left = cl * A * (rho * u**2)/2.0
    L_right = cr * A * (rho * u**2)/2.0

    #Drag force on the bird
    D = -np.sign(w) * Cd * A * (rho * w**2)/2.0

    #There is only lift if the bird is flying forwards.
    if (u > 0.0):
        F += L_left * np.cos(beta_l);
        F += L_right * np.cos(beta_r);

    F += D;
    return F;
