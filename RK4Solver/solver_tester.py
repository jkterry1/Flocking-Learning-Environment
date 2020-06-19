import numpy as np
import solver_env as solver
from bird import Bird
import time


t = 20.0
h = 0.01
n = (int)(t/h)
N = 1

birds = [Bird(z = 50.0, u = 1.0 , p = 0.0)]
env = solver.raw_env(N = len(birds), h = h, birds = birds)
env.reset()

tik = time.time()

for i in range(n):
    for agent in env.agent_order:
        a = (0.0, [0.0, 0.0, 0.0])
        # if i % 100 == 0:
        #     a = (10.0, [0.0, 0.0, 0.0])
        obs = env.step(action = a)
        #print(obs)
        #print()

tok = time.time()
print("time: ", tok-tik)
print("steps: ", n)
env.plot()
