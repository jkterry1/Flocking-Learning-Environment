import os
os.environ['SDL_AUDIODRIVER'] = 'dsp'

import sys
import gym
import random
import numpy as np

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.utils import try_import_tf
from ray.rllib.env import PettingZooEnv
# from sisl_games.pursuit import pursuit
#from pettingzoo.sisl import pursuit_v0 as game_env
import flocking_env as game_env
from supersuit import normalize_obs_v0, agent_indicator_v0, flatten_v0, frame_skip_v0, delay_observations_v0
#from supersuit import frame_skip

# for APEX-DQN
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

tf1, tf, tfv = try_import_tf()


class MLPModelV2(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name="my_model"):
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        # Simplified to one layer.
        input = tf.keras.layers.Input(obs_space.shape, dtype=obs_space.dtype)
        output = tf.keras.layers.Dense(num_outputs, activation=None)
        self.base_model = tf.keras.models.Sequential([input, output])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        return self.base_model(input_dict["obs"]), []

if __name__ == "__main__":
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN
    env_name = 'birds-v0'
    trial_name = '10birds_500frames_32workers'

    methods = ["A2C", "ADQN", "PPO", "RDQN"]

    assert len(sys.argv) == 2, "Input the learning method as the second argument"
    method = sys.argv[1]
    assert method in methods, "Method should be one of {}".format(methods)

    def env_creator(args):
        env = game_env.env(N=10)
        #env = normalize_obs_v0(env, -1000, 1000)
        #env = delay_observations_v0(env, 100)
        env = frame_skip_v0(env, 500)
        env = agent_indicator_v0(env)
        env = flatten_v0(env)
        return env

    register_env(env_name, lambda config: PettingZooEnv(env_creator(config)))

    test_env = PettingZooEnv(env_creator({}))
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    ModelCatalog.register_custom_model("MLPModelV2", MLPModelV2)
    def gen_policyV2(i):
        config = {
            "model": {
                "custom_model": "MLPModelV2",
            },
            "gamma": 0.99,
        }
        return (None, obs_space, act_space, config)
    policies = {"policy_0": gen_policyV2(0)}
    # for all methods
    policy_ids = list(policies.keys())

    if method == "A2C":
        tune.run(
            "A2C",
            name="A2C",
            stop={"episodes_total": 60000},
            checkpoint_freq=10,
            local_dir="~/A2C_results/"+trial_name,
            config={

                # Enviroment specific
                "env": "birds-v0",

                # General
                "log_level": "ERROR",
                "num_gpus": 2,
                "num_workers": 8,
                "num_envs_per_worker": 8,
                "compress_observations": False,
                "sample_batch_size": 20,
                "train_batch_size": 512,
                "gamma": .99,

                "lr_schedule": [[0, 0.0007],[20000000, 0.000000000001]],

                # Method specific

                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
            },
        )

    elif method == "ADQN":
        # APEX-DQN
        tune.run(
            "APEX",
            name="ADQN",
            stop={"episodes_total": 60000},
            checkpoint_freq=10,
            local_dir="~/ray_results_pz/"+env_name,
            config={

                # Enviroment specific
                "env": "pursuit",

                # General
                "log_level": "INFO",
                "num_gpus": 1,
                "num_workers": 4,
                "num_envs_per_worker": 8,
                "learning_starts": 1000,
                "buffer_size": int(1e5),
                "compress_observations": True,
                # "sample_batch_size": 20,
                "train_batch_size": 512,
                "gamma": .99,

                # Method specific

                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
            },
        )

    elif method == "DQN":
        # plain DQN
        tune.run(
            "DQN",
            name="DQN",
            stop={"episodes_total": 60000},
            checkpoint_freq=10,
            local_dir="~/ray_results_pz/"+env_name,
            config={
                # Enviroment specific
                "env": "pursuit",
                # General
                "log_level": "ERROR",
                "num_gpus": 1,
                "num_workers": 8,
                "num_envs_per_worker": 8,
                "learning_starts": 1000,
                "buffer_size": int(1e5),
                "compress_observations": True,
                "sample_batch_size": 20,
                "train_batch_size": 512,
                "gamma": .99,
                # Method specific
                "dueling": False,
                "double_q": False,
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
            },
        )

    elif method == "IMPALA":
        tune.run(
            "IMPALA",
            name="IMPALA",
            stop={"episodes_total": 60000},
            checkpoint_freq=10,
            local_dir="~/ray_results_pz/"+env_name,
            config={

                # Enviroment specific
                "env": "pursuit",

                # General
                "log_level": "ERROR",
                "num_gpus": 1,
                "num_workers": 8,
                "num_envs_per_worker": 8,
                "compress_observations": True,
                "sample_batch_size": 20,
                "train_batch_size": 512,
                "gamma": .99,

                "clip_rewards": True,
                "lr_schedule": [[0, 0.0005],[20000000, 0.000000000001]],

                # Method specific

                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
            },
        )

    elif method == "PPO":
        tune.run(
            "PPO",
            name="PPO",
            stop={"episodes_total": 50000},
            checkpoint_freq=1,
            local_dir="~/PPO_Results/"+trial_name,
            config={

                # Enviroment specific
                "env": "birds-v0",

                # General
                "log_level": "ERROR",
                "num_gpus": 2,
                "num_workers": 16,
                "num_envs_per_worker": 8,
                "compress_observations": False,
                "gamma": .99,


                "lambda": 0.95,
                "kl_coeff": 0.5,
                "clip_rewards": True,
                "clip_param": 0.1,
                "vf_clip_param": 10.0,
                "entropy_coeff": 0.01,
                "train_batch_size": 5000,
                # "sample_batch_size": 100,
                "sgd_minibatch_size": 500,
                "num_sgd_iter": 10,
                "batch_mode": 'truncate_episodes',
                "vf_share_layers": True,

                # Method specific

                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
            },
        )

    # psuedo-rainbow DQN
    elif method == "RDQN":
        tune.run(
            "DQN",
            name="RDQN",
            stop={"episodes_total": 60000},
            checkpoint_freq=10,
            local_dir="~/ray_results_pz/"+env_name,
            config={

                # Enviroment specific
                "env": "pursuit",

                # General
                "log_level": "ERROR",
                "num_gpus": 1,
                "num_workers": 8,
                "num_envs_per_worker": 8,
                "learning_starts": 1000,
                "buffer_size": int(1e5),
                "compress_observations": True,
                "sample_batch_size": 20,
                "train_batch_size": 512,
                "gamma": .99,

                # Method specific
                "num_atoms": 51,
                "dueling": True,
                "double_q": True,
                "n_step": 2,
                "batch_mode": "complete_episodes",
                "prioritized_replay": True,

                # # alternative 1
                # "noisy": True,
                # alternative 2
                "parameter_noise": True,

                # based on expected return
                "v_min": 0,
                "v_max": 1500,

                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
            },
        )
