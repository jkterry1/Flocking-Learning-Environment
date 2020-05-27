import numpy as np
import matplotlib.pyplot as plt

def dydt_grav(t, y, i, args):
    v0, a = args
    v0 = v0[i]
    a = a[i]
    return np.array([v0[j] + a[j] * t for j in range(0,3)])

def plot(T, Y, N):
    legend = []
    for i in range(0, N):
        plt.plot(T, [row[0] for row in Y[i]])
        plt.plot(T, [row[1] for row in Y[i]])
        plt.plot(T, [row[2] for row in Y[i]])
        x = "Bird {}: x".format(i)
        y = "Bird {}: y".format(i)
        z = "Bird {}: z".format(i)
        legend += [x,y,z]
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Bird Positions')
    plt.legend(legend)
    plt.show()


def one_bird_update(dydt, dydt_args, t0, y, h, i):
    k1 = h * dydt(t0, y, i, dydt_args)
    k2 = h * dydt(t0 + 0.5 * h, y + 0.5 * k1, i, dydt_args)
    k3 = h * dydt(t0 + 0.5 * h, y + 0.5 * k2, i, dydt_args)
    k4 = h * dydt(t0 + h, y + k3, i, dydt_args)

    return y + (1.0 / 6.0)*(k1 + (2 * k2) + (2 * k3) + k4)

# Finds value of y at a given t using step size h and initial value y0 at t0.
def bird_solver(dydt, dydt_args, t0, y0, t, h, N):
    n = (int)((t - t0)/h)
    y = y0
    #Y holds the y values for every bird
    Y = [[] for i in range(0,N)]
    T = np.arange(0, t, h)
    #go through all timesteps
    for _ in range(1, n + 1):
        #update each agent
        for i in range(0, N):
            y[i] = one_bird_update(dydt, dydt_args, t0, y[i], h, i)
            Y[i].append(np.copy(y[i]))
        t0 = t0 + h

    plot(T, Y, N)
    return Y

# Running the solver
t0 = 0
N = 2

#Initial Values
x0 = np.array([[0.0, 0.0, 0.0] for i in range(0, N)])

#constants
v0 = np.array([[5, 10, 0], [2, 20, 0]])
a = np.array([[0.0, -9.8, 0.0] for i in range (0, N)])

t = 2
h = 0.2
bird_solver(dydt_grav, (v0, a), t0, x0, t, h, N)
