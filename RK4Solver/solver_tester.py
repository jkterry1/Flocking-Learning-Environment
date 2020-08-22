import numpy as np
import solver_env as solver
from bird import Bird
import time


t = 10.0
h = 0.01
n = (int)(t/h)
N = 10

if __name__ == "__main__":
    tik = time.time()

    env = solver.env(N = 10, h = h)
    env.reset()
    done = False
    for i in range(n):
        if not done:
            for i in range(N):
                if not done:
                    #env.birds[agent].u = 7.1271
                    #env.birds[agent].print_bird(env.birds[agent])
                    a = np.zeros(5)
                    # if env.birds[agent].u < 7.1271:
                    #     a[0] = 2.0
                    obs = env.step(a)
                    reward, done, info = env.last()
                    #print("bird ", i, " ", obs)
                else:
                    break

        else:
            break
    print("timesteps ", i)
    tok = time.time()
    print("Time: ", tok - tik)
    print("Steps: ", n)
    print("Time per step: ", (tok-tik)/n)
    print("Steps per second: ", n/(tok-tik))

    env.render()
