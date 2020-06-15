import numpy as np
import solver_env as solver
from bird import Bird


t = 20.0
h = 0.01
n = (int)(t/h)
N = 1

birds = [Bird(z = 50.0, u = 0.0 , p = 3.0)]
env = solver.raw_env(N = len(birds), h = h, birds = birds)
env.reset()

for i in range(n):
    for agent in env.agent_order:
        a = (4.0, [0.0, 0.0, 0.0])
        # if i % 100 == 0:
        #     a = (10.0, [0.0, 0.0, 0.0])
        obs = env.step(action = a)
        print(obs)
env.plot()
