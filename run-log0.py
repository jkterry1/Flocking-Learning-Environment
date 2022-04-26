import numpy as np
import fle.flocking_env as flocking_env
import matplotlib.pyplot as plt
import time
import random


#env params
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

    N = 9

    env = flocking_env.raw_env(N = N,
                                h= 1/hz,
                                energy_reward=energy_reward_per_j,
                                forward_reward=distance_reward_per_m,
                                crash_reward=crash_reward,
                                LIA=True,
                                action_logging=True,)

    #file input
    action_file = "trial_22_-200_actions.csv"
    file = open(action_file)

    #agent actions:
    actions = {str(i):[] for i in range(N)}
    for line in file:
        id, action, reward = line.split(",")
        a = action.split(" ")
        action = []
        for c in a:
            if c != '[' and c != '' and c != ' ' and c != ']':
                if c[0] == '[':
                    action.append(c[1:])
                elif c[-1] == ']':
                    action.append(c[:-1])
                else:
                    action.append(c)
        action = np.array(action)
        action = action.astype(np.float)
        actions[id].append(action)



    env.reset()
    done = False
    obs, reward, done, info = env.last()

    env.reset()
    for agent in env.agent_iter():
        obs, reward, done, info = env.last()

        a = None
        if not done:
            if len(actions[env.agent_selection[-1]]) == 0:
                a = [0.0, 0.5, 0.5, 0.5, 0.5]
            else:
                a = actions[env.agent_selection[-1]].pop(0)
        env.step(a)

    #log birds for unity rendering
    env.log_birds(action_file+'_birds.csv')
    env.log_actions(action_file+'_actions.csv')

    env.plot_birds()
    #env.plot_values()

if __name__ == "__main__":
    run()
