import numpy as np
import flocking_env
import matplotlib.pyplot as plt
import plotting
#from v_policy import basic_flying_policy
import time
import random


t = 60.0
h = 0.001
n = (int)(t/h)
LIA = False

plt.plot([0],[0])
def run():
    #gliding - birds = [solver.make_bird(z=200.0, u=18.0)]
    u = 15.0
    z = 400.0

    birds = [flocking_env.make_bird(y = 0.0, x = 0.0, z=z, u = u),
            flocking_env.make_bird(y = 1.25, x = 1.0, z=z, u = u),
            flocking_env.make_bird(y = 2.5, x = 0.0, z=z, u = u)]

    #birds = [flocking_env.make_bird(y = 0.0, x = 0.0, z=z, u = u, p = 2.0)]

    # birds = [flocking_env.make_bird(y = 0.0, x = 0.0, z=z, u = u),
    #         flocking_env.make_bird(y = 1.25, x = 1.0, z=z, u = u),
    #         flocking_env.make_bird(y = 2.5, x = 2.0, z=z, u = u),
    #         flocking_env.make_bird(y = 3.75, x = 3.0, z=z, u = u),
    #         flocking_env.make_bird(y = 5.0, x = 2.0, z=z, u = u),
    #         flocking_env.make_bird(y = 6.25, x = 1.0, z=z, u = u),
    #         flocking_env.make_bird(y = 7.5, x = 0.0, z=z, u = u)]

    N = len(birds)
    env = flocking_env.raw_env(t = t, thrust_limit = 50.0, N = N, LIA = LIA, bird_inits = birds, log=True)


    env.reset()
    done = False
    obs, reward, done, info = env.last()
    steps = 0

    energies = {i:0.0 for i in env.agents}
    flaps = {i:0 for i in env.agents}

    #time measurement:
    start = time.time()
    for _ in range(500):
        env.reset()
        print("reset ---------------------------------------")
        for agent in env.agent_iter():
            obs, reward, done, info = env.last()
            energies[env.agent_selection] += reward
            steps += 1
            # a = [0.0,0.5,0.5,0.5,0.5]
            a = None
            if not done:
                a = [0.0,0.0,0.0,0.0,0.0]
                if a[0] > 0:
                    flaps[env.agent_selection] += 1
                a = np.array(a)
            #a = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
            #a[0] = basic_flying_policy(obs)

            # print(env.agent_selection)
            # print("y: ", obs[20])
            # print("x: ", obs[19])

            env.step(a)

    #print(energies)
    # print(flaps)

    #env.log_birds()
    #env.log_vortices()
    #env.plot_birds()
    # env.plot_values()

if __name__ == "__main__":
    run()
