import numpy as np
import fle.flocking_env as flocking_env
import matplotlib.pyplot as plt
#from v_policy import basic_flying_policy
import time
import random
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecTransposeImage



n_evaluations = 20
n_agents = 9
n_envs = 4
total_energy_j = 46000
total_distance_m = 870
hz = 500
crash_reward = -10
episodes = 300
nerve_impulse_hz = 200
reaction_frames = 0
n_timesteps = hz * 60 * n_agents * episodes
distance_reward_per_m = 100 / total_distance_m
energy_reward_per_j = -10 / total_energy_j
skip_frames = int(hz / nerve_impulse_hz)



env = flocking_env.parallel_env(
    N=n_agents,
    h=1 / hz,
    energy_reward=energy_reward_per_j,
    forward_reward=distance_reward_per_m,
    crash_reward=crash_reward,
    LIA=True,
)

env = ss.delay_observations_v0(env, reaction_frames)
env = ss.frame_skip_v0(env, skip_frames)



env.reset()
done = False
#obs, reward, done, info = env.last()
steps = 0

energies = {i:0.0 for i in env.agents}
#flaps = {i:0 for i in env.agents}
rew = 0
#time measurement:
start = time.time()
print('start')
obs = [0]*20
for j in range(1):
    env.reset()
    while not done:
        for agent in env.agents:
            #print('step')
            #obs, reward, done, info = env.last()
            #print(agent, ' ', obs[10],' ', obs[11],' ', obs[12], ' ', obs[13])
            # print()
            #energies[env.agent_selection] += reward
            steps += 1
            a = None
            if not done:
                w = obs[16]
                a = [0.0, 0.5, 0.5, 0.5, 0.5]
                if w < 0.5:
                    a = [1.0, 0.5, 0.5, 0.5, 0.5]
                #a = [1.0, 0.379186245, 0.90589035, 0.05797791, 1.0]
                #a = [1.0, 0.51, 0.51, 0.5, 0.5]
                #a = [1.0, random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
                #a = np.array(a)
                #a = [a]*9
                a = np.array(a)
                a = {agent:a for agent in env.agents}
                #print(a)
            obs, reward, done, info = env.step(a)
            if agent in done:
                done = done[agent]
            else:
                done = True
            if not done:
                obs = obs[agent]
                reward = reward[agent]
                rew += reward
    filename = 'bird_log_'+str(j)
    #env.log_birds(filename)
    #env.plot_birds()

print(rew/n_agents)
# print(flaps)

#env.log_birds('bird_log_1.csv')
#env.log_vortices()
#env.plot_birds()
#env.plot_values()

if __name__ == "__main__":
    run()
