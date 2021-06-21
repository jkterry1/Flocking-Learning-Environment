import numpy as np
#import solver_env as solver
import flocking_env
import matplotlib.pyplot as plt
import time
import plotting
from v_policy import basic_flying_policy
from v_policy import fly_within_range
#from v_policy import v_policy


t = 60.0
h = 0.001
n = (int)(t/h)
LIA = True

plt.plot([0],[0])
def run():
    #gliding - birds = [solver.make_bird(z=200.0, u=18.0)]
    u = 15.0
    z = 100.0

    # birds = [flocking_env.make_bird(y = 0.0, x = 0.0, z=z, u = u),
    #         flocking_env.make_bird(y = 1.25, x = 1.0, z=z, u = u),
    #         flocking_env.make_bird(y = 2.5, x = 0.0, z=z, u = u)]

    birds = [flocking_env.make_bird(y = 0.0, x = 0.0, z=z, u = u),
            flocking_env.make_bird(y = 1.25, x = 1.0, z=z, u = u),
            flocking_env.make_bird(y = 2.5, x = 2.0, z=z, u = u),
            flocking_env.make_bird(y = 3.75, x = 1.0, z=z, u = u),
            flocking_env.make_bird(y = 5.0, x = 0.0, z=z, u = u)]

    N = len(birds)
    env = flocking_env.raw_env(t = t, thrust_limit = 50.0, N = N, LIA = LIA, bird_inits = birds, log=True)
    env.reset()
    done = False
    obs, reward, done, info = env.last()
    steps = 0
    energies = {i:0.0 for i in env.agents}

    flaps = {i:0 for i in env.agents}
    for i in range(n):
        #print(i)
        if not done:
            for j in range(N):
                if not done:
                    energies[env.agent_selection] += reward
                    steps += 1
                    a = [0.0,0.5,0.5,0.5,0.5]
                    a[0] = basic_flying_policy(obs)
                    #a[0] = fly_within_range(obs)
                    #a = v_policy(3, env.agent_selection, obs)
                    if a[0] > 0:
                        #print("flap")
                        flaps[env.agent_selection] += 1
                    # print(env.agent_selection)
                    # print("y: ", obs[20])
                    # print("x: ", obs[19])

                    env.step(np.array(a))
                    obs, reward, done, info = env.last()
                else:
                    break
        else:
            # env.reset()
            done = False
            break

    print(energies)
    print(flaps)

    #env.log_birds()
    #env.log_vortices()
    env.plot_birds()
    env.plot_values()

if __name__ == "__main__":
    run()
