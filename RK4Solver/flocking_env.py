from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from gym import spaces
import numpy as np
import DiffEqs as de
from bird import Bird
from flock import Flock
import plotting
import csv

def env(**kwargs):
    env = raw_env(**kwargs)
    #env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-100)
    #env = wrappers.AssertOutOfBoundsWrapper(env)
    #env = wrappers.OrderEnforcingWrapper(env)
    #env = wrappers.NanNoOpWrapper(env, np.zeros(5), "executing the 'do nothing' action.")
    return env


class raw_env(AECEnv):

    def __init__(self,
                 N = 10,
                 h = 0.001,
                 t = 0.0,
                 birds = None,
                 LIA = False,
                 filename = "episode_1.csv"
                 ):

        self.flock = Flock(
                     N = N,
                     h = h,
                     t = t,
                     birds = birds,
                     LIA = LIA)
        self.h = h
        self.t = t
        self.N = N
        self.num_agents = N

        self.total_vortices = 0.0
        self.total_dist = 0.0

        self.episodes = 0

        self.energy_punishment = 2.0
        self.forward_reward = 5.0
        self.crash_reward = -100.0

        self.agents = range(self.N)

        self.agent_order = list(self.agents)
        self._agent_selector = agent_selector(self.agent_order)

        limit = 0.01
        action_space = spaces.Box(low = np.array([0.0, -limit, -limit, -limit, -limit]),
                                        high = np.array([10.0, limit, limit, limit, limit]))
        self.action_spaces = {i: action_space for i in self.agents}
        self.action_space = action_space

        low = -10000.0 * np.ones(22 + 6*min(self.N-1, 7),)
        high = 10000.0 * np.ones(22 + 6*min(self.N-1, 7),)
        observation_space = spaces.Box(low = low, high = high, dtype = np.float64)
        self.observation_spaces = {i: observation_space for i in self.agents}
        self.observation_space = observation_space

        self.data = []
        self.infos = {i:{} for i in self.agents}


    def step(self, action, observe = True):
        noise = 0.01 * np.random.random_sample((5,))
        #action = noise + action
        self.flock.update_bird(action, self.agent_selection)

        # reward calculation
        done, reward = self.flock.get_reward(action, self.agent_selection)
        self.rewards[self.agent_selection] = reward
        self.dones = {i:done for i in self.agents}

        # if we have moved through one complete timestep
        if self.agent_selection == self.agents[-1]:
            if self.steps % 10 == 0:
                self.flock.update_vortices(self.steps)

            self.steps += 1
            self.t = self.t + self.h

        self.agent_selection = self._agent_selector.next()

        #self.print_bird(bird, action)
        if observe:
            obs = self.observe(self.agent_selection)
            return obs


    def observe(self, agent):
        return self.flock.get_observation(agent)


    def reset(self, observe = True):
        self.episodes +=1

        self.agent_order = list(self.agents)
        self._agent_selector.reinit(self.agent_order)
        self.agent_selection = self._agent_selector.reset()

        self.flock.reset()

        self.rewards = {i: 0 for i in self.agents}
        self.dones = {i:False for i in self.agents}
        self.steps = 0

        if observe:
            return self.observe(self.agent_selection)


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
