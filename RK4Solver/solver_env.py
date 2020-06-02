from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from pettingzoo.utils.env import AECEnv
from gym import spaces
import numpy as np
from Plotting import plot
import DiffEqs as de

def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.NanNoOpWrapper(env, 26**2 * 2, "executing the 'do nothing' action.")
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):

    def __init__(self,
                 initial_vals = None,
                 ddt_args = None,
                 N = 1,
                 h = 1,
                 t = 0
                 ):
        self.h = h
        self.t = t
        self.N = N

        self.agents = ["bird_{}".format(i) for i in range(N)]
        self.agent_order = list(self.agents)
        self._agent_selector = agent_selector(self.agent_order)

        self.dims = ['x', 'y', 'z']
        self.vars = ['x', 'v', 'L']

        # maps variable -> diffeq
        self.diffeqs = {'x': de.dydt_grav, 'v': de.dvdt, 'L': de.dLdt}

        # maps agent -> variable -> arguments
        if ddt_args is None:
            ddt_args = {agent:{{} for var in self.vars} for agent in self.agents}
        self.ddt_args = ddt_args

        #maps agent -> variable -> current value
        if initial_vals is None:
            initial_vals = {agent:{var: [0.0 for dim in self.dims] for var in self.vars} for agent in self.agents}
        self.initial_vals = initial_vals

    def step(self, action, observe = True):
        agent = self.agent_selection
        #update each variable: x, v, L
        for var in self.vars:

            y = self.old_vals[agent][var]
            args = self.ddt_args[agent][var]
            ddt = self.diffeqs[var]

            #perform one timestep update to y
            step = self.one_bird_update(ddt, y, args)
            self.vals[agent][var] = step
            #store the values after making a single timestep
            self.Y[var][agent].append(np.copy(step))

        # if we have moved through one complete timestep
        if agent == self.agents[-1]:
            self.t = self.t + self.h
            self.T.append(self.t)
            self.old_vals = {agent:{var: np.copy(self.vals[agent][var]) for var in self.vars} for agent in self.agents}

        self.agent_selection = self._agent_selector.next()

    def observe():
        return


    def reset(self, observe = True):
        self.dones = {i: False for i in self.agents}

        self.agent_order = list(self.agents)
        self._agent_selector.reinit(self.agent_order)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {i: 0 for i in self.agents}

        self.vals = {agent:{var: np.copy(self.initial_vals[agent][var]) for var in self.vars} for agent in self.agents}
        self.old_vals = {agent:{var: np.copy(self.vals[agent][var]) for var in self.vars} for agent in self.agents}

        #Y holds the vals for each agent, varible -> agent -> values
        self.Y = {var:{agent:[] for agent in self.agents} for var in self.vars}
        self.T = []

    def one_bird_update(self, ddt, y, args):
        t0 = self.t
        h = self.h
        k1 = h * ddt(t0, y, self.agent_selection, args)
        k2 = h * ddt(t0 + 0.5 * h, y + 0.5 * k1, self.agent_selection, args)
        k3 = h * ddt(t0 + 0.5 * h, y + 0.5 * k2, self.agent_selection, args)
        k4 = h * ddt(t0 + h, y + k3, self.agent_selection, args)

        return y + (1.0 / 6.0)*(k1 + (2 * k2) + (2 * k3) + k4)
