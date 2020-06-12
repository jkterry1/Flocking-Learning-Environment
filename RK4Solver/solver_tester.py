import numpy as np
import solver_env as solver
from Plotting import plot
import DiffEqs as de



t0 = 0.0
t = 2.0
h = 0.2
n = (int)((t - t0)/h)
N = 1

env = solver.raw_env(N = N, h = h, t = t0)
env.reset()

for _ in range(n):
    for agent in env.agent_order:
        env.step(action = 0)
env.plot()
