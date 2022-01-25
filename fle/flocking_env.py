from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
from gym import spaces
import numpy as np
from gym.utils import seeding, EzPickle
import magent  # test to throw error if I screw up virtual env config during experiments
from . import plotting
import csv
import flocking_cpp
import random


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.ClipOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def make_bird(x=0., y=0., z=0., u=0., v=0., w=0., p=0., q=0., r=0., theta=0., phi=0., psi=0.):
    return flocking_cpp.BirdInit(x, y, z, u, v, w, p, q, r, theta, phi, psi)


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv, EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 N=10,
                 h=0.001,
                 t=60.0,
                 energy_reward=-2.0,
                 forward_reward=5.0,
                 crash_reward=-100.0,
                 bird_inits=None,
                 LIA=False,
                 action_logging=False,
                 vortex_update_frequency=100,
                 log=False,
                 num_neighbors=7,
                 derivatives_in_obs=True,
                 thrust_limit=100.0,
                 wing_action_limit_beta=0.08,
                 wing_action_limit_alpha=0.01,
                 random_seed=None,
                 include_vortices=True
                 ):
        EzPickle.__init__(self,
                          N,
                          h,
                          t,
                          energy_reward,
                          forward_reward,
                          crash_reward,
                          bird_inits,
                          LIA,
                          action_logging,
                          vortex_update_frequency,
                          log,
                          num_neighbors,
                          derivatives_in_obs,
                          thrust_limit,
                          wing_action_limit_beta,
                          wing_action_limit_alpha,
                          random_seed,
                          include_vortices
                          )
        '''
        N: number of birds (if bird_inits is None)
        h: seconds per frame (step size)
        t: maximum seconds per episode
        energy_reward: the reward for a bird using energy (negative to incentivize limiting energy use)
        forward_reward: the reward for a bird moving forward
        crash_reward: the reward for a bird crashing into another bird or the ground
        bird_inits: initial positions of the birds (None for default random sphere)
        LIA: boolean choice to include Local approximation for vortice movement
        bird_filename: the file you want to log the bird states in
        vortex_filename: the file you want to log the vortex states in
        vortex_update_frequency: Period of adding new points on the vortex line.
        log: True/False whether to log all values or not
        num_neighbors: How many neightbors each bird can see
        derivatives_in_obs: True if derivatives should be included in the observation.
        thrust_limit: The forward thrust limit of the bird (in Newtons)
        wing_action_limit: The limit on how far a wing can rotate (in degrees)
        random_seed: The random seed for noise
        '''

        self.seed(random_seed)

        # default birds are spaced 3m apart, 50m up,
        # and have an initial velocity of 5 m/s forward
        if bird_inits is None:
            bird_inits = [make_bird(z=100.0, y=3.0*i, u=15.0) for i in range(N)]

        self.t = t
        self.h = h
        self.N = len(bird_inits)

        self.energy_reward = energy_reward
        self.forward_reward = forward_reward
        self.crash_reward = crash_reward
        self.include_vortices = include_vortices

        self.bird_inits = bird_inits
        if self.bird_inits is not None:
            assert self.N == len(self.bird_inits)
        self.max_frames = int(t/h)
        self.num_neighbors = num_neighbors
        self.vortex_update_frequency = vortex_update_frequency

        self.agents = [f"b_{i}" for i in range(self.N)]
        self._agent_idxs = {b: i for i, b in enumerate(self.agents)}
        self.possible_agents = self.agents[:]

        self._agent_selector = agent_selector(self.agents)

        self.LIA = LIA
        self.derivatives_in_obs = derivatives_in_obs

        self.thrust_limit = thrust_limit
        self.wing_action_limit_beta = wing_action_limit_beta
        self.wing_action_limit_alpha = wing_action_limit_alpha

        '''
        Action space is a 5-vector where each index represents:
        0: Thrust, a forward push on the bird
        1: alpha rotation of left wing (in degrees)
        2: beta rotation of left wing (in degrees)
        3: alpha rotation of right wing (in degrees)
        4: beta rotation of right wing (in degrees)
        '''
        self.action_spaces = {i: spaces.Box(low=np.zeros(5).astype(np.float32),
                                         high=np.ones(5).astype(np.float32)) for i in self.agents}
        '''
        Observation space is a vector with
        If including derivatives, 20 dimensions for the current bird's state and
        9 dimensions for each of the birds the current bird can observe.
        If not including derivatives, 14 dimensions for the current bird's
        state and 6 dimensions for each bird the current bird can observe.

        Bird's state observations:
        0-2:    Force on the bird in each direction (fu, fv, fw)
        3-5:    Torque on the bird in each direction (Tu, Tv, Tw)
        6:      Bird's height (z)
        7-9:  Bird's orientation (phi, theta, psi)
        10-11:  Left wing orientation (alpha, beta)
        12-13:  Right wing orientation (alpha, beta)
        (The following dimensions are only included if derivatives are included)
        14-16:    Bird's velocity in each direction (u, v, w)
        17-19:  Bird's angular velocity in each direction (p, q, r)

        Following this, starting at observation 20, there will be
        9-dimension vectors for each bird the current bird can observe.
        Each of these vectors contains
        0-2:    Relative position to the current bird
        3-5:    Other bird's orientation relative to current bird (phi, theta, psi)
        (The following dimension is only included if derivatives are included)
        6-8:    Relative velocity to the current bird
        '''
        if self.derivatives_in_obs:
            low = np.zeros(20 + 9*min(self.N-1, self.num_neighbors),).astype(np.float32)
            high = np.ones(20 + 9*min(self.N-1, self.num_neighbors),).astype(np.float32)
        else:
            low = np.zeros(14 + 6*min(self.N-1, self.num_neighbors),).astype(np.float32)
            high = np.ones(14 + 6*min(self.N-1, self.num_neighbors),).astype(np.float32)
        observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_spaces = {i: observation_space for i in self.agents}

        self.action_logging = action_logging
        self.action_log = []
        self.log = log

    def step(self, action):
        if self.dones[self.agent_selection]:
            # this function handles agent termination logic like
            # checking that the action is None and removing the agent from the agents list
            return self._was_done_step(action)

        self._cumulative_rewards[self.agent_selection] = 0
        # denormalize the action
        denorm_action = np.zeros(5)
        denorm_action[0] = action[0] * self.thrust_limit
        denorm_action[1] = action[1] * 2.0 * self.wing_action_limit_alpha - self.wing_action_limit_alpha
        denorm_action[2] = action[2] * 2.0 * self.wing_action_limit_beta - self.wing_action_limit_beta
        denorm_action[3] = action[3] * 2.0 * self.wing_action_limit_alpha - self.wing_action_limit_alpha
        denorm_action[4] = action[4] * 2.0 * self.wing_action_limit_beta - self.wing_action_limit_beta

        # update the current bird's properties and generate it's vortices
        # for the next time step using the given action
        self.simulation.update_bird(denorm_action, self._agent_idxs[self.agent_selection])

        done, reward = self.simulation.get_done_reward(denorm_action, self._agent_idxs[self.agent_selection])

        # log the agent, action, and reward from this time step
        if self.action_logging is True:
            self.action_log.append([self.agent_selection[2:], denorm_action, reward])

        self._clear_rewards()
        self.rewards[self.agent_selection] = reward

        # End this episode if it has exceeded the maximum time allowed.
        if self.steps >= self.max_frames:
            done = True
        self.done = done or self.done

        '''
        If we have cycled through all of the birds for one time step,
        increase total steps performed by 1 and
        update the vortices if we are on a vortex update timestep
        (vortices update once every self.vortex_update_frequency steps)
        '''
        if self.agent_selection == self.agents[-1]:
            if self.steps % self.vortex_update_frequency == 0:
                if self.include_vortices:
                    self.simulation.update_vortices(self.vortex_update_frequency)
                if(self.log):
                    self.log_vortices(self.vortex_filename)
            self.steps += 1
            self.dones = {agent: done or self.done for agent in self.agents}

        # move on to the next bird
        self.agent_selection = self._agent_selector.next()

        self._accumulate_rewards()  # this function adds everything in the rewards dict into the _cumulative_rewards dict

    def observe(self, agent):
        return self.simulation.get_observation(self._agent_idxs[agent], self.num_neighbors)

    def reset(self):
        # creates c++ environment with initial birds
        self.simulation = flocking_cpp.Flock(self.N, self.h, self.t,
                                            self.energy_reward, self.forward_reward,
                                            self.crash_reward, self.bird_inits,
                                            self.LIA, self.derivatives_in_obs,
                                            self.thrust_limit, self.wing_action_limit_alpha,
                                            self.wing_action_limit_beta, self.random_seed, self.include_vortices)

        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.simulation.reset()

        self._cumulative_rewards = {name: 0. for name in self.agents}
        self.rewards = {i: 0 for i in self.agents}
        self.dones = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        self.done = False
        self.steps = 0

    def render(self, mode='human', plot_vortices=False):
        # replace with something functional or entirely remove for log based rendering
        birds = self.simulation.get_birds()
        plotting.plot_values(birds, show=True)
        plotting.plot_birds(birds, plot_vortices=plot_vortices, show=True)

    def close(self):
        plotting.close()

    '''
    Plots a graph of various values for the birds over time:
    1st graph:  position
    2nd graph:  angular velocity
    3rd graph:  angle
    4th graph:  velocity
    '''
    def plot_values(self):
        plotting.plot_values(self.simulation.get_birds())

    '''
    Plots the bird and vortex positions in a 3D graph.
    There are yellow lines representing the paths of the centers of the birds,
    and red points that represent the centers of active vortices.

    if the vortices parameter is true, then arrows showing the rotation of
    the vortices will be plotted as well.
    '''
    def plot_birds(self, vortices=False):
        plotting.plot_birds(self.simulation.get_birds(), plot_vortices=vortices)

    '''
    Writes the vortex states to a csv file that can go into the Unity animation.
    Each time a new vortex point is added, all currently active vortices are logged.
    Each line in the file contains 8 values:
    0:      the current time
    1-3:    The vortex's position (x, y, z)
    4-6:    The voretex's orientation (theta, phi, psi)
    7:      The current strength of the vortex
    '''
    def log_vortices(self, file_name):
        file = open(file_name, "w")
        wr = csv.writer(file)
        birds = self.simulation.get_birds()
        time = self.steps * self.h
        for bird in birds:
            for vortex in bird.VORTICES_LEFT:
                state = [time, vortex.pos[0], vortex.pos[1], vortex.pos[2], vortex.theta, vortex.phi, vortex.psi, vortex.gamma]
                wr.writerow(state)
            for vortex in bird.VORTICES_RIGHT:
                state = [time, vortex.pos[0], vortex.pos[1], vortex.pos[2], vortex.theta, vortex.phi, vortex.psi, vortex.gamma]
                wr.writerow(state)

    '''
    Writes the bird states to a csv file that can be used in the Unity animation
    Each line in the file contains 12 values:
    0:      The bird's ID (birds are numbered 0 to N-1)
    1:      The current time
    2-4:    The bird's position (x, y, z)
    5-7:    The bird's orientation (phi, theta, psi)
    8-9:    The bird's left and right wing alpha angles
    10-11:  The bird's left and right wing beta angles
    '''
    def log_birds(self, file_name):
        file = open(file_name, "w")
        wr = csv.writer(file)
        birds = self.simulation.get_birds()
        for ID, bird in enumerate(birds):
            state = []
            for i in range(len(bird.X)):
                if i % 10 == 0:
                    time = i*self.h
                    state = [ID, time, bird.X[i], bird.Y[i], bird.Z[i], bird.PHI[i], bird.THETA[i], bird.PSI[i], bird.ALPHA_L[i], bird.ALPHA_R[i], bird.BETA_L[i], bird.BETA_R[i]]
                    wr.writerow(state)

    def log_actions(self, file_name):
        file = open(file_name, "w")
        writer = csv.writer(file)

        if self.action_logging is False:
            print("action logging must be enabled at initialization to use this function")
        else:
            for row in self.action_log:
                writer.writerow(row)

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def seed(self, seed=None):
        # C++ unsigned int size is 4 bytes
        self.random_seed = seeding.create_seed(seed, max_bytes=4)
