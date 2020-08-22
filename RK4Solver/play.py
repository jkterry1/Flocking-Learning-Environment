from solver_env import env as custom_env
import ray
from ray.tune.registry import register_trainable, register_env
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.dqn import ApexTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.a3c.a2c import A2CTrainer
from ray.rllib.models import Model, ModelCatalog
from ray.rllib.env import PettingZooEnv
import os
import pickle
import numpy as np
from ray.rllib.models import ModelCatalog
from supersuit.aec_wrappers import flatten
import tensorflow as tf
ray.init()

# 2 cases: APEX_DQN, RAINBOW_DQN or everything else
MLPv2methods = ["APEX_DQN", "RAINBOW_DQN"]
METHOD = ""   # "RAINBOW_DQN" # "APEX_DQN" # ""

class MLPModel(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        last_layer = tf.layers.dense(
                input_dict["obs"], 400, activation=tf.nn.relu, name="fc1")
        last_layer = tf.layers.dense(
            last_layer, 300, activation=tf.nn.relu, name="fc2")
        output = tf.layers.dense(
            last_layer, num_outputs, activation=None, name="fc_out")
        return output, last_layer

ModelCatalog.register_custom_model("MLPModel", MLPModel)

env_name = "birds-v0"
# path should end with checkpoint-<> data file
checkpoint_path = "/home/caroline/PPO_Results/500frames_32workers/PPO/PPO_birds-v0_0_2020-08-15_18-40-42cp9ii2ll/checkpoint_1"
Trainer = PPOTrainer

# TODO: see ray/rllib/rollout.py -- `run` method for checkpoint restoring

# register env -- For some reason, ray is unable to use already registered env in config
def env_creator(args):
    env = custom_env()
    env = flatten(env)
    return env

env = env_creator(1)
register_env(env_name, env_creator)

# get the config file - params.pkl
with open("/home/caroline/PPO_Results/500frames_32workers/PPO/PPO_birds-v0_0_2020-08-15_18-40-42cp9ii2ll/params.pkl", "rb") as f:
    config = pickle.load(f)

#ray.init()

register_env(env_name, lambda config: PettingZooEnv(env_creator(config)))
RLAgent = Trainer(env=env_name, config=config)
RLAgent.restore(checkpoint_path)

# init obs, action, reward
observations = env.reset()
rewards, action_dict = {}, {}
for agent_id in env.agent_ids:
    assert isinstance(agent_id, int), "Error: agent_ids are not ints."
    # action_dict = dict(zip(env.agent_ids, [np.array([0,1,0]) for _ in range(len(env.agent_ids))])) # no action = [0,1,0]
    rewards[agent_id] = 0

totalReward = 0
done = False
# action_space_len = 3 # for all agents

# TODO: extra parameters : /home/miniconda3/envs/maddpg/lib/python3.7/site-packages/ray/rllib/policy/policy.py

iteration = 0
while not done:
    action_dict = {}
    # compute_action does not cut it. Go to the policy directly
    for agent_id in env.agent_ids:
        # print("id {}, obs {}, rew {}".format(agent_id, observations[agent_id], rewards[agent_id]))
        action, _, _ = RLAgent.get_policy("policy_0").compute_single_action(observations[agent_id], prev_reward=rewards[agent_id]) # prev_action=action_dict[agent_id]
        # print(action)
        action_dict[agent_id] = action

    observations, rewards, dones, info = env.step(action_dict)
    env.render()
    totalReward += sum(rewards.values())
    done = any(list(dones.values()))
    print("iter:", iteration, sum(rewards.values()))
    iteration += 1

env.close()
print("done", done, totalReward)
