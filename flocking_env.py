from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from gym import spaces
import numpy as np
import plotting
import csv
import flocking_cpp

def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.ClipOutOfBoundsWrapper(env)
    env = wrappers.NanNoOpWrapper(env, np.zeros(5), "executing the 'do nothing' action.")
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def make_bird(x=0., y=0., z=0., u=0., v=0., w=0., p=0., q=0., r=0., theta=0., phi=0., psi=0.):
    return flocking_cpp.BirdInit(x, y, z, u, v, w, p, q, r, theta, phi, psi)

class raw_env(AECEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 N = 10,
                 h = 0.001,
                 t = 0.0,
                 bird_inits=None,
                 LIA = False,
                 filename = "episode_1.csv"
                 ):

        if bird_inits is None:
            bird_inits = [make_bird(z = 50.0, y = 3.0*i, u=5.0) for i in range(N)]

        # creates c++ flocking object with initial bird values and hyperparameters
        self.flock = flocking_cpp.Flock(
                     N,
                     h,
                     t,
                     bird_inits,
                     LIA)

        self.h = h
        self.t = t
        self.N = N

        self.agents = [f"b_{i}" for i in range(self.N)]
        self._agent_idxs = {b:i for i, b in enumerate(self.agents)}
        self.possible_agents = self.agents[:]

        self.agent_order = list(self.agents)
        self._agent_selector = agent_selector(self.agent_order)

        limit = 0.01
        action_space = spaces.Box(low = np.array([0.0, -limit, -limit, -limit, -limit]),
                                        high = np.array([10.0, limit, limit, limit, limit]))
        self.action_spaces = {i: action_space for i in self.agents}
        self.action_space = action_space

        low = -10000.0 * np.ones(22 + 6*min(self.N-1, 7),)
        high = 10000.0 * np.ones(22 + 6*min(self.N-1, 7),)
        observation_space = spaces.Box(low = low, high = high, dtype = np.float32)
        self.observation_spaces = {i: observation_space for i in self.agents}
        self.observation_space = observation_space

        self.data = []
        self.infos = {i:{} for i in self.agents}

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        cur_agent = self.agent_selection
        noise = 0.01 * np.random.random_sample((5,))
        #action = noise + action
        self.flock.update_bird(action, self._agent_idxs[self.agent_selection])

        # reward calculation
        done, reward = self.flock.get_reward(action, self._agent_idxs[self.agent_selection])
        self._clear_rewards()
        self.rewards[self.agent_selection] = reward
        self.dones = {i:done for i in self.agents}
        # print(done)
        # print(reward)
        # for bird in self.flock.get_birds():
        #     print(bird.z)

        # if we have moved through one complete timestep
        if self.agent_selection == self.agents[-1]:
            vortex_update_frequency = 10
            if self.steps % vortex_update_frequency == 0:
                self.flock.update_vortices(vortex_update_frequency)

            self.steps += 1
        self.agent_selection = self._agent_selector.next()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()
        self._dones_step_first()


    def observe(self, agent):
        return self.flock.get_observation(self._agent_idxs[agent])


    def seed(self, seed=None):
        pass


    def reset(self):
        self.agent_order = list(self.agents)
        self._agent_selector.reinit(self.agent_order)
        self.agent_selection = self._agent_selector.reset()

        self.flock.reset()

        self._cumulative_rewards = {name: 0. for name in self.agents}
        self.rewards = {i: 0 for i in self.agents}
        self.dones = {i:False for i in self.agents}
        self.steps = 0


    def render(self, mode='human', plot_vortices=False):
        birds = self.flock.get_birds()
        plotting.plot_values(birds, show = True)
        plotting.plot_birds(birds, plot_vortices = plot_vortices, show = True)


    def close(self):
        plotting.close()


    def log(self):
        birds = self.flock.get_birds()
        for bird in birds:
            state = []
            #[ID, x, y, z, phi, theta, psi, aleft, aright, bleft, bright]
            ID = self.agents.index(self.agent_selection)
            time = self.steps * self.h
            state = [ID, time, bird.x, bird.y, bird.z, bird.phi, bird.theta, bird.psi, bird.alpha_l, bird.alpha_r, bird.beta_l, bird.beta_r]
            self.data.append(state)
            # writing the data into the file
            if np.any(self.dones):
                wr = csv.writer(self.file)
                wr.writerows(self.data)
