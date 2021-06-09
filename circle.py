
import numpy as np
import math
from numpy.linalg import norm
import matplotlib.pyplot as plt


class Vortex:
    def __init__(self, x, y, z, phi, theta, psi, sign):
        self.sign = sign;
        self.C = 15.0;
        self.min_vel = 1.0;
        self.max_r = 1.0;
        self.dist_travelled = 0.0;

        self.phi = phi;
        self.theta = theta;
        self.psi = psi;

        self.pos = [x,y,z];
        self.X = self.pos[0];
        self.Y = self.pos[1];
        self.Z = self.pos[2];

        self.gamma_0 = 10.0;
        self.gamma = self.gamma_0;
        self.decay_std_dev = 10.0;
        self.decay_time_mean = 1000.0 * 3.0 * self.decay_std_dev;
        self.decay_exponent = 2.0;
        self.t_decay = 1000.0 * 1.0;
        self.t = 0.0;

        self.decaying = False

        #velocity of the vortex
        vel = 0.0;

        #self.gamma = 15.0;

        #core represents the radius of the vortex's "core", the center of the
        #vortex where no air is moving to prevent divide by zero issue.
        self.core = 0.05;

        self.B = []




def drdt(gamma, epsilon, b, theta):
    return (gamma/(4.0* np.pi * epsilon)) * b * math.tan(theta/2.0);


def take_vortex_time_step(B, pos, sign, gamma, epsilon, b, theta, h):
    k1 = h * drdt(gamma, epsilon, b, theta)
    k2 = h * drdt(gamma, epsilon, b, theta)
    k3 = h * drdt(gamma, epsilon, b, theta)
    k4 = h * drdt(gamma, epsilon, b, theta)
    delta = (1.0 / 6.0)*(k1 + (2.0 * k2) + (2.0 * k3) + k4)
    B.append(delta[2])
    return pos + delta

def update_vortex_positions(vortices, h):
    L = len(vortices)
    assert len(vortices)%L == 0
    new_vals = []
    for i in range(0, len(vortices)):
        #r_n
        pos = [vortices[i%L].X, vortices[i%L].Y, vortices[i%L].Z]

        #r_n-1
        pos_minus = np.array([vortices[(i-1)].X, vortices[(i-1)].Y, vortices[(i-1)].Z])

        #r_n+1
        pos_plus = np.array([vortices[(i+1)%L].X, vortices[(i+1)%L].Y, vortices[(i+1)%L].Z])

        #r_n+2
        pos_plus2 = np.array([vortices[(i+2)%L].X, vortices[(i+2)%L].Y, vortices[(i+2)%L].Z])

        #tangent vectors
        t_minus = pos - pos_minus
        t = pos_plus - pos
        t_plus = pos_plus2 - pos_plus

        #length of tangent vectors
        l_t_minus = np.linalg.norm(t_minus)
        l_t = np.linalg.norm(t)


        #checking the limits of arccos
        if (np.dot(t_minus, t)/(l_t_minus * l_t)) <= -0.99:
            print("theta out of bounds")
            theta = np.arccos(-1.0)

        elif (np.dot(t_minus, t)/(l_t_minus * l_t)) >= 0.99:
            print("theta out of bounds")
            theta = np.arccos(1.0)

        else:
            #theta = np.arccos(np.dot(t_minus, t)/(l_t_minus * l_t))
            theta = np.arccos(np.dot(t_minus, t)/(l_t_minus * l_t))
            #print("theta ",theta)

        #normal vector
        #n = (np.cross(t, t_plus)/(1 + np.dot(t, t_plus))) - (np.cross(t_minus, t)/(1 + np.dot(t_minus, t)))
        n = (t - t_minus)
        n = n/norm(n)
        #binormal vector
        b = np.cross(t, n);

        #normalize n and b, (protecting against divide by zero)
        if (norm(n) > 0.0):
            n = n/np.linalg.norm(n)

        if (norm(b) > 0.0):
            b = b/np.linalg.norm(b)

        #strength of the vortex
        if i > 0:
            gamma = vortices[i %L].gamma
        else:
            gamma = vortices[i].gamma
        #distance from last point
        epsilon = l_t_minus
        assert epsilon>0.00001

        sign = 1.0
        #vortex's new position after LIA calculation
        new_vals.append(take_vortex_time_step(vortices[i].B, pos, sign, gamma, epsilon, b, theta, h))

        #Update vortex strength based on decay equation
        vor = vortices[i]
        vor.t += 1
        if vor.decaying:
            #print(" val ", (float(vor.t_decay)/float(vor.t)))
            vor.gamma = vor.gamma_0 * pow((float(vor.t_decay)/float(vor.t)), float(vor.decay_exponent))

        #Replace with probability if we decide not to make it deterministic
        else:
            if vor.t > vor.t_decay:
                vor.decaying = True

        #distance from last point
        epsilon = l_t_minus

    for i in range(len(vortices)):
        #update distance travelled and position for this time step
        vortices[i%L].X = new_vals[i][0];
        vortices[i%L].Y = new_vals[i][1];
        vortices[i%L].Z = new_vals[i][2];
        vortices[i%L].pos = new_vals[i];


def plot_vortices(vortices):
    Xnew = []
    Ynew = []
    Znew = []
    for vortex in vortices:
        Xnew.append(vortex.pos[0])
        Ynew.append(vortex.pos[1])
        Znew.append(vortex.pos[2])
    print("a: ", np.sqrt((Xnew[0]-Xnew[len(Xnew)//2])**2 + (Ynew[0]-Ynew[len(Xnew)//2])**2))
    plt.plot(Xnew, Ynew)


locs = [(np.sqrt(3.0)/2.0, 0.5), (np.sqrt(2.0)/2.0, np.sqrt(2.0)/2.0), (0.5, np.sqrt(3.0)/2.0)]
signs = [(1.0,1.0), (-1.0,-1.0), (-1.0, 1.0), (1.0, -1.0)]

X = []
Y = []
Z = []
vortices = []
vortices2 = []
vortices3 = []
vortices4 = []
vortices5 = []
vortices6 = []
vortices7 = []
vortices8 = []
vortices9 = []
vortices10 = []

N = 20
rad = 1.0
for i in range(0, N):
    x = np.cos((2.0*np.pi/float(N)) * float(i))
    y = np.sin((2.0*np.pi/float(N)) * float(i))
    z = 0.0
    vortices.append(Vortex(x,y,0.0,0.0,0.0,0.0,-1.0))
    X.append(x)
    Y.append(y)
    Z.append(z)
plt.plot(X, Y)

rad = 1.5
for i in range(0, N):
    x = np.cos((2.0*np.pi/float(N)) * float(i)) * rad
    y = np.sin((2.0*np.pi/float(N)) * float(i)) * rad
    z = 0.0
    vortices2.append(Vortex(x,y,0.0,0.0,0.0,0.0,-1.0))

rad = 2.0
for i in range(0, N):
    x = np.cos((2.0*np.pi/float(N)) * float(i)) * rad
    y = np.sin((2.0*np.pi/float(N)) * float(i)) * rad
    z = 0.0
    vortices3.append(Vortex(x,y,0.0,0.0,0.0,0.0,-1.0))

rad = 2.5
for i in range(0, N):
    x = np.cos((2.0*np.pi/float(N)) * float(i)) * rad
    y = np.sin((2.0*np.pi/float(N)) * float(i)) * rad
    z = 0.0
    vortices4.append(Vortex(x,y,0.0,0.0,0.0,0.0,-1.0))

rad = 3.0
for i in range(0, N):
    x = np.cos((2.0*np.pi/float(N)) * float(i)) * rad
    y = np.sin((2.0*np.pi/float(N)) * float(i)) * rad
    z = 0.0
    vortices5.append(Vortex(x,y,0.0,0.0,0.0,0.0,-1.0))

rad = 3.5
for i in range(0, N):
    x = np.cos((2.0*np.pi/float(N)) * float(i)) * rad
    y = np.sin((2.0*np.pi/float(N)) * float(i)) * rad
    z = 0.0
    vortices6.append(Vortex(x,y,0.0,0.0,0.0,0.0,-1.0))

rad = 4.0
for i in range(0, N):
    x = np.cos((2.0*np.pi/float(N)) * float(i)) * rad
    y = np.sin((2.0*np.pi/float(N)) * float(i)) * rad
    z = 0.0
    vortices7.append(Vortex(x,y,0.0,0.0,0.0,0.0,-1.0))

rad = 4.5
for i in range(0, N):
    x = np.cos((2.0*np.pi/float(N)) * float(i)) * rad
    y = np.sin((2.0*np.pi/float(N)) * float(i)) * rad
    z = 0.0
    vortices8.append(Vortex(x,y,0.0,0.0,0.0,0.0,-1.0))

rad = 5.0
for i in range(0, N):
    x = np.cos((2.0*np.pi/float(N)) * float(i)) * rad
    y = np.sin((2.0*np.pi/float(N)) * float(i)) * rad
    z = 0.0
    vortices9.append(Vortex(x,y,0.0,0.0,0.0,0.0,-1.0))

rad = 5.5
for i in range(0, N):
    x = np.cos((2.0*np.pi/float(N)) * float(i)) * rad
    y = np.sin((2.0*np.pi/float(N)) * float(i)) * rad
    z = 0.0
    vortices10.append(Vortex(x,y,0.0,0.0,0.0,0.0,-1.0))


s = 3
t = int(1000.0)
for i in range(s):
    for _ in range(t):
        update_vortex_positions(vortices, 0.001)
        update_vortex_positions(vortices2, 0.001)
        update_vortex_positions(vortices3, 0.001)
        update_vortex_positions(vortices4, 0.001)
        update_vortex_positions(vortices5, 0.001)
        update_vortex_positions(vortices6, 0.001)
        update_vortex_positions(vortices7, 0.001)
        update_vortex_positions(vortices8, 0.001)
        update_vortex_positions(vortices9, 0.001)
        update_vortex_positions(vortices10, 0.001)
    plot_vortices(vortices)
legend = [str(i) + " seconds" for i in range(0, s+1)]

plt.show()
X = [1,1.5,2,2.5,3,3.5,4,4.5,5, 5.5]
Y = [vortices[0].B[0],vortices2[0].B[0],vortices3[0].B[0],vortices4[0].B[0],vortices5[0].B[0],
    vortices6[0].B[0],vortices7[0].B[0],vortices8[0].B[0],vortices9[0].B[0],vortices10[0].B[0]]
plt.plot(X, Y)
plt.xlabel("Ring Radius (m)")
plt.ylabel("Forward velocity (m/s)")
plt.title("Relationship Between Ring Radius and Forward Velocity")
plt.show()
