from solver_env import env as _env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np


class env(MultiAgentEnv):

    metadata = {'render.modes': ['human']}

    def __init__(self, N, *args, **kwargs):
        super(env, self).__init__()
        self.env = _env(*args, **kwargs)
        self.N = N
        self.num_agents = self.N
        self.agent_ids = list(range(self.num_agents))
        # spaces
        self.act_dims = [5 for i in range(self.N)]
        self.n_act_agents = self.act_dims[0]
        #self.action_space_dict = dict(zip(self.agent_ids, self.env.action_space))
        self.action_space_dict = dict(zip(self.agent_ids, [self.env.action_space for i in range(N)]))
        self.observation_space_dict = dict(zip(self.agent_ids, [self.env.observation_space for i in range(N)]))
        self.steps = 0

        self.reset()

    def convert_to_dict(self, list_of_list):
        return dict(zip(self.agent_ids, list_of_list))

    def reset(self):
        obs = self.env.reset()
        #print("OBS RESET ", obs)
        self.steps = 0
        return self.convert_to_dict([obs for _ in range(self.num_agents)])

    def close(self):
        self.env.close()

    def render(self):
        self.env.render()

    def step(self, action_dict):
        # unpack actions
        action_list = np.array([np.zeros(5) for _ in range(self.num_agents)])

        for agent_id in self.agent_ids:
            if np.any(np.isnan(action_dict[agent_id])):
                action_dict[agent_id] = np.zeros(5)

            action_list[agent_id] = action_dict[agent_id]

        observation, reward, done, info = self.env.step(action_list[0])
        observation = observation
        #print("OBSERVATION ", observation)

        if self.steps >= 50000:
            done = {i:True for i in range(self.num_agents)}

        observation_dict = self.convert_to_dict([observation for _ in range(self.num_agents)])
        reward_dict = reward
        info_dict = info
        done_dict = done
        done_dict["__all__"] = done[0]

        self.steps += 1

        return observation_dict, reward_dict, done_dict, info_dict
