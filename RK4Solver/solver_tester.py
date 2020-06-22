import numpy as np
import solver_env as solver
from bird import Bird
import time


t = 2.0
h = 0.1
n = (int)(t/h)
N = 1
games = 1

tik = time.time()

birds = [Bird(z = 5.0, x = 1.0, u = 5.0 , p = 0.0), Bird(z = 5.0, y = .75, u = 5.0, p = 0.0),  Bird(z = 5.0, y = -.75, u = 5.0, p = 0.0)]
env = solver.raw_env(N = len(birds), h = h, birds = birds)
env.reset()

env.reset()
for i in range(n):
    for agent in env.agent_order:
        a = (0.0, [0.0, 0.0, 0.0])
        # if i % 10 == 0:
        #     a = (10.0, [0.0, 0.0, 0.0])
        obs = env.step(action = a)
        #print(obs)
env.plot_values()
env.plot_birds(False)
tok = time.time()
print("Time: ", tok - tik)
print("Steps: ", n)
