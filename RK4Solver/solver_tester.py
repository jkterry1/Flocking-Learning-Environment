import numpy as np
import solver_env as solver
from bird import Bird


t = 10.0
h = 0.1
n = (int)(t/h)
N = 1

birds = [Bird(z = 1.0, u = 1.0 , p = 0.0)]
env = solver.raw_env(N = len(birds), h = h, birds = birds)
env.reset()

for i in range(n):
    for agent in env.agent_order:
        a = (0.0, [0.0, 0.0, 0.0])
        # if i % 100 == 0:
        #     a = (10.0, [0.0, 0.0, 0.0])
        obs = env.step(action = a)
        print(obs)
env.plot()
