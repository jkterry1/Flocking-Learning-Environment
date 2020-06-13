from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from pettingzoo.utils.env import AECEnv
from gym import spaces
import numpy as np
import DiffEqs as de
import bird
from bird import Bird
import matplotlib.pyplot as plt

def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.NanNoOpWrapper(env, 0, "executing the 'do nothing' action.")
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):

    def __init__(self,
                 N = 1,
                 h = 0.1,
                 t = 0.0
                 ):
        self.h = h
        self.t = t
        self.N = N

        self.agents = ["bird_{}".format(i) for i in range(N)]
        self.birds = {agent: Bird(p = 3.0, u = 5.0) for agent in self.agents}
        self.agent_order = list(self.agents)
        self._agent_selector = agent_selector(self.agent_order)



    def step(self, action, observe = True):
        bird = self.birds[self.agent_selection]

        bird.update(self.t, self.h)

        # if we have moved through one complete timestep
        if self.agent_selection == self.agents[-1]:
            self.t = self.t + self.h

        self.agent_selection = self._agent_selector.next()

    def observe(self):
        return

    def plot(self):
        for bird in self.birds:
            bird = self.birds[bird]
            t = np.arange(0, self.t, self.h)

            plt.subplot(311)
            plt.title('Position')
            plt.plot(t, bird.X)
            plt.plot(t, bird.Y)
            plt.plot(t, bird.Z)
            plt.legend(['x', 'y', 'z'])

            plt.subplot(312)
            plt.title('Velocity')
            plt.plot(t, bird.U)
            plt.plot(t, bird.V)
            plt.plot(t, bird.W)
            plt.legend(['u', 'v', 'w'])

            plt.subplot(313)
            plt.title('Angle')
            plt.plot(t, bird.THETA)
            plt.plot(t, bird.PHI)
            plt.plot(t, bird.PSI)
            plt.legend(['theta', 'phi', 'psi'])
            plt.show()



    def reset(self, observe = True):
        self.dones = {i: False for i in self.agents}

        self.agent_order = list(self.agents)
        self._agent_selector.reinit(self.agent_order)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {i: 0 for i in self.agents}
