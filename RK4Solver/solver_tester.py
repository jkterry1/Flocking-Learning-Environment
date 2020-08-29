import numpy as np
#import solver_env as solver
import flocking_env as solver
from bird import Bird
import time


t = 10.0
h = 0.001
n = (int)(t/h)
N = 10

def run():
    tik = time.time()
    z = 0.01
    birds = [Bird(z=90.0, y=0.6, x=-1.0, u=0.5, p = 5.0)]
    env = solver.env(N = 1, birds = birds)
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

    avg = env.total_dist/env.total_vortices
    print("average distance travelled of a vortex: ", avg)

    env.render(plot_vortices = False)

if __name__ == "__main__":
    run()
