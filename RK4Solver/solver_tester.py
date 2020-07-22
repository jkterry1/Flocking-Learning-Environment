import numpy as np
import solver_env as solver
from bird import Bird
import time


t = 20.0
h = 0.001
n = (int)(t/h)
N = 1

tik = time.time()

birds = [Bird(z = 50.0, x = 0.0, y = 0.0, u = 2.0), Bird(z = 50.0, x = -1.0, y = -1.0, u = 2.0)]
        #Bird(z = 50.0, x = -1.0, y = -1.0, u = 2.0),
        #Bird(z = 50.0, x = -1.0, y = -1.0, u = 2.0), Bird(z = 50.0, x = -1.0, y = -1.0, u = 2.0),
        #Bird(z = 50.0, x = -1.0, y = -1.0, u = 2.0), Bird(z = 50.0, x = -1.0, y = -1.0, u = 2.0)]
env = solver.raw_env(N = len(birds), h = h, birds = birds, filename = "test1.csv")
env.reset()
done = False
for i in range(n):
    if not done:
        for agent in env.agent_order:
            if not env.dones[agent]:
                #env.birds[agent].u = 7.1271
                #env.birds[agent].print_bird(env.birds[agent])
                a = [  30.0, \
                     np.random.uniform(-0.01, 0.01), np.random.uniform(-0.01, 0.01), \
                    np.random.uniform(-0.01, 0.01), np.random.uniform(-0.01, 0.01)]
                a = np.zeros(5)
                # if env.birds[agent].u < 7.1271:
                #     a[0] = 2.0
                obs = env.step(action = a)
                #print(obs)
            else:
                done = True
                break
    else:
        break
print("timesteps ", i)
tok = time.time()
print("Time: ", tok - tik)
print("Steps: ", n)
print("Time per step: ", (tok-tik)/n)
print("Steps per second: ", n/(tok-tik))

env.plot_values(show = True)
env.plot_birds(show = True)
