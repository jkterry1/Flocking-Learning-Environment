from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from gym import spaces
import numpy as np
import plotting
import csv
from gym.utils import seeding
import flocking_cpp


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.ClipOutOfBoundsWrapper(env)
    env = wrappers.NanNoOpWrapper(env, np.zeros(5), "executing the 'do nothing' action.")
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
                 energy_punishment=2.0,
                 forward_reward=5.0,
                 crash_reward=-100.0,
                 max_observable_birds=7,
                 bird_inits=None,
                 LIA=False,
                 filename="episode_1.csv",
                 vortex_update_frequency=10
                 ):
        '''
        N: number of birds (if bird_inits is None)
        h: seconds per frame (step size)
        t: maximum seconds per episode
        bird_inits: initial positions of the birds
        LIA: Local approximation for vortice movement
        '''
        if bird_inits is None:
            bird_inits = [make_bird(z=50.0, y=3.0*i, u=5.0) for i in range(N)]  # change this to random points in a sphere,that takes sphere size as an argument
        else:
            N = len(bird_inits)

        # creates c++ environment with initial birds
        self.simulation = flocking_cpp.Flock(N, h, t, energy_punishment, forward_reward, crash_reward, bird_inits, LIA)
        self.seed()

        self.h = h
        self.N = N
        self.max_frames = int(t/h)
        self.max_observable_birds = max_observable_birds  # nearest birds that can be observed
        self.vortex_update_frequency = vortex_update_frequency

        self.agents = [f"b_{i}" for i in range(self.N)]
        self._agent_idxs = {b: i for i, b in enumerate(self.agents)}
        self.possible_agents = self.agents[:]

        self.agent_order = list(self.agents)
        self._agent_selector = agent_selector(self.agent_order)

        limit = 0.01
        #action_space =   # first is thrust (needs units), others need to be labeled and dimensioned (why 4?)
        self.action_spaces = {i: spaces.Box(low=np.array([0.0, -limit, -limit, -limit, -limit]),
                                        high=np.array([10.0, limit, limit, limit, limit])) for i in self.agents}

        # They're giant because there's position, so there's no clear limit. Smaller ones should be used for things other than that. Comment needed with each element of vector
        low = -10000.0 * np.ones(22 + 6*min(self.N-1, self.max_observable_birds),)
        high = 10000.0 * np.ones(22 + 6*min(self.N-1, self.max_observable_birds),)
        observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_spaces = {i: observation_space for i in self.agents}

        self.data = []  # used for logging

        self.infos = {i: {} for i in self.agents}

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)  # please explain what this function does
        cur_agent = self.agent_selection
        noise = 0.01 * self.np_random.random_sample((5,))  # this isn't used anywhere?

        self.simulation.update_bird(action, self._agent_idxs[self.agent_selection])

        done, reward = self.simulation.get_done_reward(action, self._agent_idxs[self.agent_selection])  # 1 why is action taken here 2 the logic separating this line and the last is unclear

        self._clear_rewards()
        self.rewards[self.agent_selection] = reward
        if self.steps >= self.max_frames:
            done = True
        self.dones = {agent: done for agent in self.agents}

        if self.agent_selection == self.agents[-1]:
            if self.steps % self.vortex_update_frequency == 0:
                self.simulation.update_vortices(self.vortex_update_frequency)  # why is the argument needed?
            self.steps += 1

        self.agent_selection = self._agent_selector.next()

        self._cumulative_rewards[cur_agent] = 0  # why is that needed? why can't we just call this before the above line and use self.agent_selection?
        self._accumulate_rewards()  # please explain what this function does
        self._dones_step_first()  # please explain what this function does

    def observe(self, agent):
        return self.simulation.get_observation(self._agent_idxs[agent], self.max_observable_birds)

    def reset(self):
        self.agents = self.possible_agents[:]
        self.agent_order = list(self.agents)  # why is the list() needed, and why is this as it's own variable needed?
        self._agent_selector.reinit(self.agent_order)
        self.agent_selection = self._agent_selector.reset()

        self.simulation.reset()

        self._cumulative_rewards = {name: 0. for name in self.agents}
        self.rewards = {i: 0 for i in self.agents}
        self.dones = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        self.steps = 0

    def render(self, mode='human', plot_vortices=False):
        # This function generates plots summarizing the statics of birds
        birds = self.simulation.get_birds()
        plotting.plot_values(birds, show=True)
        plotting.plot_birds(birds, plot_vortices=plot_vortices, show=True)

    def close(self):
        plotting.close()

    def log(self):
        birds = self.simulation.get_birds()
        for bird in birds:
            state = []
            # [ID, x, y, z, phi, theta, psi, aleft, aright, bleft, bright]
            ID = self.agents.index(self.agent_selection)
            time = self.steps * self.h
            # alpha and beta are wing angles for each wings, though which means what is unclear
            state = [ID, time, bird.x, bird.y, bird.z, bird.phi, bird.theta, bird.psi, bird.alpha_l, bird.alpha_r, bird.beta_l, bird.beta_r]
            self.data.append(state)
            if np.any(self.dones):
                wr = csv.writer(self.file)
                wr.writerows(self.data)
