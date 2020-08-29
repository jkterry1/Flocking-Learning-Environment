from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from gym import spaces
import numpy as np
import DiffEqs as de
from bird import Bird
import flocking_helpers as helpers
import plotting
import copy
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
                 filename = "episode_1.csv"
                 ):
        self.h = h
        self.t = t
        self.N = N
        self.num_agents = N

        self.total_vortices = 0.0
        self.total_dist = 0.0

        self.max_r = 1.0
        self.max_steps = 100.0/self.h

        self.episodes = 0

        self.energy_punishment = 2.0
        self.forward_reward = 5.0
        self.crash_reward = -100.0

        self.agents = ["bird_{}".format(i) for i in range(N)]
        if birds is None:
            birds = [Bird(z = 50.0, y = 3.0*i, u=5.0) for i in range(self.N)]
        self.starting_conditions = [copy.deepcopy(bird) for bird in birds]
        self.birds = {agent: birds[i] for i, agent in enumerate(self.agents)}
        self.agent_order = list(self.agents)
        self._agent_selector = agent_selector(self.agent_order)

        limit = 0.01
        action_space = spaces.Box(low = np.array([0.0, -limit, -limit, -limit, -limit]),
                                        high = np.array([10.0, limit, limit, limit, limit]))
        self.action_spaces = {bird: action_space for bird in self.birds}
        self.action_space = action_space

        low = -10000.0 * np.ones(22 + 6*min(self.N-1, 7),)
        high = 10000.0 * np.ones(22 + 6*min(self.N-1, 7),)
        observation_space = spaces.Box(low = low, high = high)
        self.observation_spaces = {bird: observation_space for bird in self.agents}
        self.observation_space = observation_space

        self.data = []
        self.infos = {i:{} for i in self.agents}

    def step(self, action, observe = True):
        bird = self.birds[self.agent_selection]
        noise = 0.01 * np.random.random_sample((5,))
        #action = noise + action
        helpers.update_bird(self, action)

        # reward calculation
        done, reward = helpers.get_reward(self,action)
        self.rewards[self.agent_selection] = reward
        self.dones = {i:done for i in self.agents}

        # if we have moved through one complete timestep
        if self.agent_selection == self.agents[-1]:
            if self.steps % 10 == 0:
                helpers.update_vortices(self)

            self.steps += 1
            self.positions = helpers.get_positions(self)
            self.t = self.t + self.h

        self.agent_selection = self._agent_selector.next()

        #self.print_bird(bird, action)
        if observe:
            obs = self.observe(self.agent_selection)
            return obs

    def observe(self, agent):
        obs = helpers.get_observation(self, agent)
        return obs

    def reset(self, observe = True):
        self.episodes +=1

        self.agent_order = list(self.agents)
        self._agent_selector.reinit(self.agent_order)
        self.agent_selection = self._agent_selector.reset()

        self.old_birds = {bird: copy.deepcopy(self.birds[bird]) for bird in self.birds}
        birds = [copy.deepcopy(bird) for bird in self.starting_conditions]
        self.birds = {agent: birds[i] for i, agent in enumerate(self.agents)}
        self.positions = helpers.get_positions(self)

        self.rewards = {i: 0 for i in self.agents}
        self.dones = {i:False for i in self.agents}
        self.steps = 0

        if observe:
            obs = self.observe(self.agent_selection)
            return obs

    def render(self, mode='human', plot_vortices=False):
        plotting.plot_values(self.birds, show = True)
        plotting.plot_birds(self.birds, plot_vortices = plot_vortices, show = True)

    def close(self):
        plotting.close()

    def print_bird(self, bird, action = []):
        print('-----------------------------------------------------------------------')
        print(self.agent_selection)
        print("thrust, al, ar, bl, br \n", action)
        print("x, y, z: \t\t", [bird.x, bird.y, bird.z])
        print("u, v, w: \t\t", [bird.u, bird.v, bird.w])
        print("phi, theta, psi: \t", [bird.phi, bird.theta, bird.psi])
        print("p, q, r: \t\t", [bird.p, bird.q, bird.r])
        print("alpha, beta (left): \t", [bird.alpha_l, bird.beta_l])
        print("alpha, beta (right): \t", [bird.alpha_r, bird.beta_r])
        print("Fu, Fv, Fw: \t\t", bird.F)
        print("Tu, Tv, Tw: \t\t", bird.T)
        print("VFu, VFv, VFw: \t\t", bird.vortex_force_u, bird.vortex_force_v, bird.vortex_force_w)
        print("VTu, VTv, VTw: \t\t", bird.vortex_torque_u, bird.vortex_torque_v, bird.vortex_torque_w)
        print()

    def log(self, bird):
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
