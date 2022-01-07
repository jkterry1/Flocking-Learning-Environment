import numpy as np
import fle.flocking_env as flocking_env
import matplotlib.pyplot as plt
import time
import random


t = 60.0
hz = 1000
h=1/hz
n = (int)(t/h)
LIA = False

total_energy_j = 46000
total_distance_m = 870
distance_reward_per_m = 100/total_distance_m
energy_reward_per_j = -10/total_energy_j
crash_reward = -10


plt.plot([0],[0])
def run():

    u = 15.0
    z = 2000.0

    #custom starting state for each agent
    birds = [
            flocking_env.make_bird(y = 0.0, z=z, u = u),
            flocking_env.make_bird(y = 3.0, z=z, u = u),
            flocking_env.make_bird(y = 6.0, z=z, u = u)]
    N = len(birds)

    env = flocking_env.raw_env(N = N,
                                h= 1/hz,
                                energy_reward=energy_reward_per_j,
                                forward_reward=distance_reward_per_m,
                                crash_reward=crash_reward,
                                bird_inits = birds,
                                LIA=True,
                                action_logging=True,)

    env.reset()
    done = False
    obs, reward, done, info = env.last()

    env.reset()
    for agent in env.agent_iter():
        obs, reward, done, info = env.last()

        a = None
        if not done:
            #do nothing
            a = [0.0, 0.5, 0.5, 0.5, 0.5]
            a = np.array(a)
        env.step(a)

    #log birds for unity rendering
    #env.log_actions('action_log_1.csv')

    env.plot_birds()
    env.plot_values()

if __name__ == "__main__":
    run()
