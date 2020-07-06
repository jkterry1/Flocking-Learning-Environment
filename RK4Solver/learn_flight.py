from stable_baselines.common.env_checker import check_env
import solver_env
from stable_baselines import A2C
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines.common.callbacks import CheckpointCallback, EveryNTimesteps

env = solver_env.env()
env = DummyVecEnv([lambda: env])
#env = VecCheckNan(env, raise_exception=True)

checkpoint_on_event = CheckpointCallback(save_freq=1, save_path='./logs/')
event_callback = EveryNTimesteps(n_steps=1000000, callback=checkpoint_on_event)

model = A2C('MlpPolicy', env, verbose=1, tensorboard_log="./birds_tensorboard/")
model.learn(total_timesteps=int(2e6), tb_log_name="first_run")
model.save("a2c_birds")

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
