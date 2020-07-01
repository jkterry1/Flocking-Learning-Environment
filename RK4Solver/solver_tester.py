import numpy as np
import solver_env as solver
from bird import Bird
import time


t = 20.0
h = 0.01
n = (int)(t/h)
N = 1

tik = time.time()

birds = [Bird(z = 50.0, x = 0.5, y = 1.0, alpha_l = -np.pi/50.0, u = 10.0)]
env = solver.raw_env(N = len(birds), h = h, birds = birds)
env.reset()

for i in range(n):
    if not env.done:
        for agent in env.agent_order:
            #env.birds[agent].u = 7.1271
            a = [np.random.uniform(0.0, 5.0), \
                 np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), \
                np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)]
            a = np.zeros(5)
            # if env.birds[agent].u < 7.1271:
            #     a[0] = 2.0
            obs = env.step(action = a)
            #print(obs)
    else:
        break
print("timesteps ", i)
print("reward ", env.reward)
tok = time.time()
print("Time: ", tok - tik)
print("Steps: ", n)

env.plot_values()
env.plot_birds(False)
