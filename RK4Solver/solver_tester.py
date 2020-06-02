import numpy as np
import solver_env as solver
from Plotting import plot
import DiffEqs as de



t0 = 0
t = 2
h = 0.2
n = (int)((t - t0)/h)
N = 2

agents = ["bird_{}".format(i) for i in range(N)]

#ddt args
v0 = {'bird_0':np.array([5, 10, 0]), 'bird_1':np.array([2, 20, 0])}
a = {agent:np.array([0.0, -9.8, 0.0]) for agent in agents}
w = np.zeros(3)
# maps agent -> variable -> arguments
dydt_args = {agent:{'x':{'v0':v0[agent], 'a':a[agent]}, 'v':{'F':de.F, 'm':1.0}, 'L':{'T':de.T, 'w': w}} for agent in agents}

env = solver.raw_env(N = N, ddt_args = dydt_args, h = h, t = t0)
env.reset()

for _ in range(n):
    for agent in env.agent_order:
        env.step(action = 0)

plot(env.T, env.Y['x'], env.agents)
