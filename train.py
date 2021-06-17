from stable_baselines3 import PPO
import flocking_env
import supersuit as ss
from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first

n_evaluations = 20
n_agents = 10
n_envs = 4
n_timesteps = 15360000  # same aprox number as pistonball is 1600*10*6000=96 million- this is ~256 episodes


def image_transpose(env):
    if is_image_space(env.observation_space) and not is_image_space_channels_first(env.observation_space):
        env = VecTransposeImage(env)
    return env


env = flocking_env.parallel_env()
env = ss.color_reduction_v0(env, mode='B')
env = ss.resize_v0(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 3)
env = ss.pettingzoo_env_to_vec_env_v0(env)
env = ss.concat_vec_envs_v0(env, n_envs, num_cpus=1, base_class='stable_baselines3')
env = VecMonitor(env)

eval_env = flocking_env.parallel_env()
eval_env = ss.color_reduction_v0(eval_env, mode='B')
eval_env = ss.resize_v0(eval_env, x_size=84, y_size=84)
eval_env = ss.frame_stack_v1(eval_env, 3)
eval_env = ss.pettingzoo_env_to_vec_env_v0(eval_env)
eval_env = ss.concat_vec_envs_v0(eval_env, 1, num_cpus=1, base_class='stable_baselines3')
eval_env = VecMonitor(eval_env)

eval_freq = int(n_timesteps / n_evaluations)
eval_freq = max(eval_freq // (n_envs*n_agents), 1)

model = PPO("MlpPolicy", env, verbose=3, batch_size=64, n_steps=512, gamma=0.99, learning_rate=0.00018085932590331433, ent_coef=0.09728964435428247, clip_range=0.4, n_epochs=10, vf_coef=0.27344752686795376, gae_lambda=0.9, max_grad_norm=5)
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=eval_freq, deterministic=True, render=False)
model.learn(total_timesteps=n_timesteps, callback=eval_callback)

"""
What sticky actions to use?
What observation delay to use?
What frame stacking to use?
"""
