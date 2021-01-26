import numpy as np
#import solver_env as solver
import flocking_env as solver
import matplotlib.pyplot as plt
import time
import plotting


t = 10.0
h = 0.001
n = (int)(t/h)
N = 1
LIA = True

def run():
    z = 0.01
    birds = [solver.make_bird(z=50.0, p=2.0, u=11.0)]
    env = solver.raw_env(N = N, LIA = LIA, bird_inits = birds)
    env.reset()
    done = False

    for i in range(n):
        if not done:
            for i in range(N):
                if not done:
                    a = np.zeros(5)
                    a[0] = 1
                    obs = env.step(a)
                    done = env.dones[env.agent_selection]
                    rew = env.rewards[env.agent_selection]
                else:
                    break
        else:
            break
            env.reset()
            done = False
    env.log_birds()
    env.log_vortices()
    env.plot_birds()
    env.plot_values()

if __name__ == "__main__":
    run()
