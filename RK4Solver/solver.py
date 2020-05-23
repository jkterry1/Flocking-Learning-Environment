import matplotlib.pyplot as plt
import numpy as np

# A sample differential equation "dy / dt = v0 + at"
def dydt_grav(t, y, args):
    v0, a = args
    return np.array([v0[i] + a[i] * t for i in range(0,3)])

# Finds value of y at a given t using step size h and initial value y0 at t0.
def runge_kutta(dydt, dydt_args, t0, y0, t, h):
    n = (int)((t - t0)/h)
    y = y0
    Y = []
    T = np.arange(0, t, h)
    for i in range(1, n + 1):
        k1 = h * dydt(t0, y, dydt_args)
        k2 = h * dydt(t0 + 0.5 * h, y + 0.5 * k1, dydt_args)
        k3 = h * dydt(t0 + 0.5 * h, y + 0.5 * k2, dydt_args)
        k4 = h * dydt(t0 + h, y + k3, dydt_args)

        y = y + (1.0 / 6.0)*(k1 + (2 * k2) + (2 * k3) + k4)
        t0 = t0 + h

        Y.append(y)

    plt.plot(T, [row[0] for row in Y])
    plt.plot(T, [row[1] for row in Y])
    plt.plot(T, [row[2] for row in Y])
    plt.xlabel('Time')
    plt.ylabel('position')
    plt.legend(['x', 'y', 'z'])
    plt.show()
    return y

# Running the solver
t0 = 0
y = np.array([0, 0, 0])
v0 = np.array([5, 10, 0])
a = np.array([0, -9.8, 0])
t = 2
h = 0.2
runge_kutta(dydt_grav, (v0, a), t0, y, t, h)
