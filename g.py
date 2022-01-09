# https://github.com/jkterry1/rl-baselines3-zoo/blob/flocking/render_optimization_policies.py
import os
import sys
from os.path import exists

import fle.flocking_env as flocking_env
import numpy as np
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecTransposeImage

#num = sys.argv[1]

n_agents = 9
n_envs = 4
total_energy_j = 24212
total_distance_m = 894
hz = 500
crash_reward = -10
episodes = 300
nerve_impulse_hz = 200
reaction_frames = 0
time = 10
n_timesteps = hz * time * n_agents * episodes
distance_reward_per_m = 100 / total_distance_m
energy_reward_per_j = -10 / total_energy_j
skip_frames = int(hz / nerve_impulse_hz)


render_env = flocking_env.env(
    N=n_agents,
    h=1 / hz,
    energy_reward=energy_reward_per_j,
    forward_reward=distance_reward_per_m,
    crash_reward=crash_reward,
    action_logging=True,
    LIA=True,
)
render_env = ss.delay_observations_v0(render_env, reaction_frames)
render_env = ss.frame_skip_v0(render_env, skip_frames)


filepath = "no_limit_optimization_policies/trial_155/best_model"

print("Loading new policy ", filepath)
model = PPO.load(filepath)

i = 0
render_env.reset()

rewards = {agent:0 for agent in render_env.agents}

for agent in render_env.agent_iter():
    observation, reward, done, _ = render_env.last()
    rewards[agent] += reward
    action = model.predict(observation, deterministic=True)[0] if not done else None
    act_str = "Agent: " + str(agent) + "\t Action: " + str(action)+"\n"
    render_env.step(action)

avg_rew = 0
for agent in rewards:
    avg_rew += rewards[agent] / n_agents
print(avg_rew)
#print("Saving vortex logs")
#render_env.unwrapped.log_vortices("./results/" + policy +"_vortices" + ".csv")
# print("Saving bird logs: ./results/" + policy + "_" +
#                                 str(avg_rew) + "_birds" + ".csv")
# render_env.unwrapped.log_birds("./results/" + policy + "_" +
#                                 str(avg_rew) + "_birds" + ".csv")
# render_env.unwrapped.log_actions("./results/" + policy + "_" +
#                                 str(avg_rew) + "_actions" + ".csv")