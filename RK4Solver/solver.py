import numpy as np
import matplotlib.pyplot as plt
from Plotting import plot
import DiffEqs as de

class Solver():
    def __init__(self,
                 initial_vals = None,
                 ddt_args = None,
                 N = 1
                 ):
        self.N = N
        self.agents = range(N)
        self.dims = ['x', 'y', 'z']
        self.vars = ['x', 'v', 'L']

        # maps variable -> diffeq
        self.diffeqs = {'x': de.dydt_grav, 'v': de.dvdt, 'L': de.dLdt}

        # maps agent -> variable -> arguments
        if ddt_args is None:
            ddt_args = {{{} for var in self.vars} for agent in self.agents}
        self.ddt_args = ddt_args

        #maps agent -> variable -> current value
        if initial_vals is None:
            initial_vals = {agent:{var: [0.0 for dim in self.dims] for var in self.vars} for agent in self.agents}
        self.vals = initial_vals


    def one_bird_update(self, ddt, t0, y, h, args, agent):
        k1 = h * ddt(t0, y, agent, args)
        k2 = h * ddt(t0 + 0.5 * h, y + 0.5 * k1, agent, args)
        k3 = h * ddt(t0 + 0.5 * h, y + 0.5 * k2, agent, args)
        k4 = h * ddt(t0 + h, y + k3, agent, args)

        return y + (1.0 / 6.0)*(k1 + (2 * k2) + (2 * k3) + k4)


    def solve(self, t0, t, h):
        n = (int)((t - t0)/h)

        #Y holds the vals for each agent, varible -> agent -> values
        Y = {var:{agent:[] for agent in self.agents} for var in self.vars}
        T = np.arange(0, t, h)

        for _ in range(n):
            old_vals = {agent:{var: np.copy(self.vals[agent][var]) for var in self.vars} for agent in self.agents}

            for agent in self.agents:
                for var in self.vars:

                    y = old_vals[agent][var]
                    args = self.ddt_args[agent][var]
                    ddt = self.diffeqs[var]

                    #perform one timestep update to y
                    step = self.one_bird_update(ddt, t0, y, h, args, agent)
                    self.vals[agent][var] = step
                    #store the values after making a single timestep
                    Y[var][agent].append(np.copy(step))

            t0 = t0 + h
            old_vals = self.vals

        #currently only plots x
        plot(T, Y['x'], self.agents)
        return Y


# Running the solver
# Right now this just solves gravity for position, with ODEs that do nothing for v and L.
# ----------- NEXT STEP: Make sure that old values are saved so future agents arent affected by already updated agent's values
if __name__ == "__main__":
    t0 = 0
    t = 2
    h = 0.2
    n = 2

    #ddt args
    v0 = [np.array([5, 10, 0]), np.array([2, 20, 0])]
    a = [np.array([0.0, -9.8, 0.0]) for i in range(0, n)]
    w = np.zeros(3)
    # maps agent -> variable -> arguments
    dydt_args = [{'x':{'v0':v0[agent], 'a':a[agent]}, 'v':{'F':de.F, 'm':1.0}, 'L':{'T':de.T, 'w': w}} for agent in range(n)]

    s = Solver(N = n, ddt_args = dydt_args)
    s.solve(t0, t, h)
