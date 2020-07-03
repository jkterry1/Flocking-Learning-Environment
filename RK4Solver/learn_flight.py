from stable_baselines.common.env_checker import check_env
import solver_env
from stable_baselines import PPO2
from stable_baselines.common.cmd_util import make_vec_env

env = solver_env.env()
env = make_vec_env(lambda: env, n_envs=4)

model = PPO2('MlpPolicy', env, verbose=1, tensorboard_log="./birds_tensorboard/")
model.learn(total_timesteps=10000, tb_log_name="first_run")

print("Done learning!")

# Test the trained agent
obs = env.reset()
n_steps = 20000
for step in range(n_steps):
  action, _ = model.predict(obs, deterministic=True)
  #print("Step {}".format(step + 1))
  print("Action: ", action)
  obs, reward, done, info = env.step(action)

  #print('obs=', obs, 'reward=', reward, 'done=', done)
  if done.any():
    # Note that the VecEnv resets automatically
    # when a done signal is encountered
    env.render(mode = 'human')
    print("Goal reached!", "reward=", reward)
    break
