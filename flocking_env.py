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
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def make_bird(x=0., y=0., z=0., u=0., v=0., w=0., p=0., q=0., r=0., theta=0., phi=0., psi=0.):
    # a comment is needed defining all of these, including their units
    return flocking_cpp.BirdInit(x, y, z, u, v, w, p, q, r, theta, phi, psi)


class raw_env(AECEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 N=10,
                 h=0.001,
                 t=60.0,
                 energy_reward=-2.0,
                 forward_reward=5.0,
                 crash_reward=-100.0,
                 max_observable_birds=7,
                 bird_inits=None,
                 LIA=False,
                 bird_filename="bird_log_1.csv",
                 vortex_filename = "vortex_log_1.csv",
                 vortex_update_frequency=100
                 ):
        '''
        N: number of birds (if bird_inits is None)
        h: seconds per frame (step size)
        t: maximum seconds per episode
        energy_reward: the reward for a bird using energy (negative to incentivize limiting energy use)
        forward_reward: the reward for a bird moving forward
        crash_reward: the reward for a bird crashing into another bird or the ground
        max_observable_birds: the number of neighboring birds a bird can see
        bird_inits: initial positions of the birds (None for default random sphere)
        LIA: boolean choice to include Local approximation for vortice movement
        bird_filename: the file you want to log the bird states in
        vortex_filename: the file you want to log the vortex states in
        vortex_update_frequency: Period of adding new points on the vortex line.
        '''

        # default birds are spaced 3m apart, 50m up,
        # and have an initial velocity of 5 m/s forward
        if bird_inits is None:
            bird_inits = [make_bird(z=50.0, y=3.0*i, u=5.0) for i in range(N)]

        self.t = t
        self.h = h
        self.N = N

        self.energy_reward = energy_reward
        self.forward_reward = forward_reward
        self.crash_reward = crash_reward

        self.bird_inits = bird_inits
        if self.bird_inits is not None:
            assert self.N == len(self.bird_inits)
        self.max_frames = int(t/h)
        self.max_observable_birds = max_observable_birds
        self.vortex_update_frequency = vortex_update_frequency

        self.agents = [f"b_{i}" for i in range(self.N)]
        self._agent_idxs = {b: i for i, b in enumerate(self.agents)}
        self.possible_agents = self.agents[:]

        self._agent_selector = agent_selector(self.agents)

        self.LIA = LIA

        # the limit on how many degrees a bird can twist or lift its wings
        # per time step.
        wing_action_limit = 0.01

        '''
        Action space is a 5-vector where each index represents:
        0: Thrust, a forward push on the bird
        1: alpha rotation of left wing (in degrees)
        2: beta rotation of left wing (in degrees)
        3: alpha rotation of right wing (in degrees)
        4: beta rotation of right wing (in degrees)
        '''
        self.action_spaces = {i: spaces.Box(low=np.array([0.0, -wing_action_limit, -wing_action_limit, -wing_action_limit, -wing_action_limit]).astype(np.float32),
                                        high=np.array([10.0, wing_action_limit, wing_action_limit, wing_action_limit, wing_action_limit]).astype(np.float32)) for i in self.agents}

        '''
        Observation space is a vector with
        22 dimensions for the current bird's state and
        6 dimensions for each of the birds the current bird can observe.

        Bird's state observations:
        0-2:    Force on the bird in each direction (fu, fv, fw)
        3-5:    Torque on the bird in each direction (Tu, Tv, Tw)
        6-8:    Bird's position in each dimension (x, y, z)
        9-11:   Bird's velocity in each direction (u, v, w)
        12-14:  Bird's angular velocity in each direction (p, q, r)
        15-17:  Bird's orientation (phi, theta, psi)
        18-19:  Left wing orientation (alpha, beta)
        20-21:  Right wing orientation (alpha, beta)

        Following this, starting at observation 22, there will be
        6-dimension vectors for each bird the current bird can observe.
        Each of these vectors contains
        0-2:    Relative position to the current bird
        3-5:    Relative velocity to the current bird
        '''
        low = -10000.0 * np.ones(22 + 6*min(self.N-1, self.max_observable_birds),).astype(np.float32)
        high = 10000.0 * np.ones(22 + 6*min(self.N-1, self.max_observable_birds),).astype(np.float32)
        observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_spaces = {i: observation_space for i in self.agents}

        self.vortex_file = open(vortex_filename, "w")
        self.bird_filename = bird_filename

    def step(self, action):
        if self.dones[self.agent_selection]:
            # this function handles agent termination logic like
            # checking that the action is None and removing the agent from the agents list
            return self._was_done_step(action)

        # update the current bird's properties and generate it's vortices
        # for the next time step using the given action
        self.simulation.update_bird(action, self._agent_idxs[self.agent_selection])

        done, reward = self.simulation.get_done_reward(action, self._agent_idxs[self.agent_selection])

        self._clear_rewards()
        self.rewards[self.agent_selection] = reward

        # End this episode if it has exceeded the maximum time allowed.
        if self.steps >= self.max_frames:
            done = True
        self.dones = {agent: done for agent in self.agents}

        '''
        If we have cycled through all of the birds for one time step,
        increase total steps performed by 1 and
        update the vortices if we are on a vortex update timestep
        (vortices update once every self.vortex_update_frequency steps)
        '''
        if self.agent_selection == self.agents[-1]:
            if self.steps % self.vortex_update_frequency == 0:
                self.simulation.update_vortices(self.vortex_update_frequency)
                if(log):
                    self.log_vortices()
            self.steps += 1

        self._cumulative_rewards[self.agent_selection] = 0

        # move on to the next bird
        self.agent_selection = self._agent_selector.next()

        self._accumulate_rewards()  # this function adds everything in the rewards dict into the _cumulative_rewards dict
        self._dones_step_first()  # this handles the agent death logic. It is necessary here, but I guess it probably should not be necessary here because there is no non-trivial death mechanics here. If you want, you can create an issue of this, and I can fix it so that this call isn't necessary.

    def observe(self, agent):
        return self.simulation.get_observation(self._agent_idxs[agent], self.max_observable_birds)

    def reset(self):
        # creates c++ environment with initial birds
        self.simulation = flocking_cpp.Flock(self.N, self.h, self.t, self.energy_reward, self.forward_reward, self.crash_reward, self.bird_inits, self.LIA)

        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.simulation.reset()

        self._cumulative_rewards = {name: 0. for name in self.agents}
        self.rewards = {i: 0 for i in self.agents}
        self.dones = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
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
        plotting.plot_birds(self.simulation.get_birds(), plot_vortices = vortices)


    '''
    Writes the vortex states to a csv file that can go into the Unity animation.
    Each time a new vortex point is added, all currently active vortices are logged.
    Each line in the file contains 8 values:
    0:      the current time
    1-3:    The vortex's position (x, y, z)
    4-6:    The voretex's orientation (theta, phi, psi)
    7:      The current strength of the vortex
    '''
    def log_vortices(self):
        file = self.vortex_file
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
    def log_birds(self):
        file = open(self.bird_filename, "w")
        wr = csv.writer(file)
        birds = self.simulation.get_birds()
        for ID, bird in enumerate(birds):
            state = []
            for i in range(len(bird.X)):
                time = i*self.h
                state = [ID, time, bird.X[i], bird.Y[i], bird.Z[i], bird.PHI[i], bird.THETA[i], bird.PSI[i], bird.ALPHA_L[i], bird.ALPHA_R[i], bird.BETA_L[i], bird.BETA_R[i]]
                wr.writerow(state)
