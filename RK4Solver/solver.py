import numpy as np
import matplotlib.pyplot as plt
from Plotting import plot
import DiffEqs as de

class Solver():
    def __init__(self,
                 initial_vals = None,
                 ddt_args = None,
                 diffeqs = None,
                 N = 1
                 ):
        self.N = N
        self.dims = ['x', 'y', 'z']
        #self.vars = ['x', 'v', 'L']
        self.vars = ['x']
        self.agents = ["agent_{}".format(i) for i in range(N)]
        if initial_vals is None:
            initial_vals = {agent:{var: [0.0 for dim in self.dims] for var in self.vars} for agent in self.agents}
        self.vals = initial_vals


    def one_bird_update(self, ddt, dydt_args, t0, y, h, i):
        k1 = h * ddt(t0, y, i, dydt_args)
        k2 = h * ddt(t0 + 0.5 * h, y + 0.5 * k1, i, dydt_args)
        k3 = h * ddt(t0 + 0.5 * h, y + 0.5 * k2, i, dydt_args)
        k4 = h * ddt(t0 + h, y + k3, i, dydt_args)

        return y + (1.0 / 6.0)*(k1 + (2 * k2) + (2 * k3) + k4)

    # Finds value of y at a given t using step size h and initial value y0 at t0.
    def solve(self, dydt, dydt_args, t0, t, h):
        n = (int)((t - t0)/h)
        #Y holds the y values for every bird
        Y = {agent: [] for agent in self.agents}
        T = np.arange(0, t, h)
        #go through all timesteps
        for _ in range(1, n + 1):
            #update each agent
            for agent in self.agents:
                for var in self.vars:
                    y = self.vals[agent][var]
                    a = self.one_bird_update(dydt, dydt_args, t0, y, h, agent)
                    self.vals[agent][var] = a
                    Y[agent].append(np.copy(a))
            t0 = t0 + h

        plot(T, Y, self.agents)
        return Y


# Running the solver
if __name__ == "__main__":
    t0 = 0
    n = 2
    #ddt args
    agents = ["agent_{}".format(i) for i in range(n)]
    v0 = {'agent_0':np.array([5, 10, 0]), 'agent_1':np.array([2, 20, 0])}
    a = {"agent_{}".format(i):np.array([0.0, -9.8, 0.0]) for i in range (0, n)}
    dydt_args = (v0, a)

    t = 2
    h = 0.2
    s = Solver(N = n)
    s.solve(de.dydt_grav, dydt_args, t0, t, h)
