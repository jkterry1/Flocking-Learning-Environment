from pettingzoo.test import api_test
from pettingzoo.test import bombardment_test
from pettingzoo.test import performance_benchmark
from pettingzoo.test import render_test
from pettingzoo.test import seed_test
from pettingzoo.test import test_save_obs
from pettingzoo.test import max_cycles_test

from pettingzoo.utils import random_demo

import time

import flocking_env
import random
import numpy as np


def modified_average_total_reward(env, max_episodes=10):
    total_reward = 0
    total_steps = 0
    evaluation_steps = [0]

    for episode in range(max_episodes):

        env.reset()
        for agent in env.agent_iter():
            obs, reward, done, _ = env.last()
            total_reward += reward
            total_steps += 1
            if done:
                action = None
            elif isinstance(obs, dict) and 'action_mask' in obs:
                action = random.choice(np.flatnonzero(obs['action_mask']))
            else:
                action = env.action_spaces[agent].sample()
            env.step(action)
        evaluation_steps.append(total_steps)
        num_episodes = episode + 1
    print("Average total reward", total_reward / num_episodes)
    print('Evaluation steps', evaluation_steps)
    print('Average Steps per episode', total_steps/num_episodes)


env = flocking_env.env()
env_fn = flocking_env.env


# api_test(env, num_cycles=1, render=False, verbose_progress=True)
# performance_benchmark(env)
# random_demo(env, cycles=1)

# seed_test(env_fn)

startTime = time.time()
modified_average_total_reward(env, max_episodes=50)
print('Rune time', time.time()-startTime)
