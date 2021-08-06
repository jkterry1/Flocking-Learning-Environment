from stable_baselines3 import PPO
import flocking_env
import supersuit as ss
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
import os

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

n_timesteps = hz*60*n_agents*episodes
print("n_timesteps: " + str(n_timesteps))
distance_reward_per_m = 100/total_distance_m
energy_reward_per_j = -10/total_energy_j
skip_frames = int(hz/nerve_impulse_hz)
print("skip_frames: " + str(skip_frames))

render_env = flocking_env.env(N=n_agents, h=1/hz, energy_reward=energy_reward_per_j, forward_reward=distance_reward_per_m, crash_reward=crash_reward, LIA=True)
render_env = ss.delay_observations_v0(render_env, reaction_frames)
render_env = ss.frame_skip_v0(render_env, skip_frames)

policies = os.listdir(os.getcwd() + '/logs/eval_callback/')

for i, policy in enumerate(policies):
    model = PPO.load(os.getcwd() + '/logs/eval_callback/' + policy)

    render_env.reset()

    while True:
        for agent in render_env.agent_iter():
            observation, _, done, _ = render_env.last()
            action = model.predict(observation, deterministic=True)[0] if not done else None
            render_env.step(action)

        render_env.unwrapped.log_vortices('./sim_logs/log_vortices_' + str(i) + '.csv')
        render_env.unwrapped.log_birds('./sim_logs/log_birds_' + str(i) + '.csv')
        break
