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
from mpl_toolkits.mplot3d import Axes3D


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
                 t = 0.0,
                 birds = None
                 ):
        self.h = h
        self.t = t
        self.N = N

        self.agents = ["bird_{}".format(i) for i in range(N)]
        if birds is None:
            birds = [Bird(z = 100.0) for agent in self.agents]
        self.birds = {agent: birds[i] for i, agent in enumerate(self.agents)}
        self.agent_order = list(self.agents)
        self._agent_selector = agent_selector(self.agent_order)



    def step(self, action, observe = True):
        bird = self.birds[self.agent_selection]
        thrust, torque = action

        bird.update(thrust, torque, self.h)

        # if we have moved through one complete timestep
        if self.agent_selection == self.agents[-1]:
            self.positions = self.get_positions()
            self.t = self.t + self.h

        self.agent_selection = self._agent_selector.next()
        if observe:
            return self.observe()

    def observe(self):
        force = self.birds[self.agent_selection].F
        torque = self.birds[self.agent_selection].T
        return [force, torque, self.positions]

    def plot(self):
        for bird in self.birds:
            bird = self.birds[bird]
            t = np.arange(0, self.t+0*self.h, self.h)

            # plt.subplot(211)
            # plt.title('Position')
            # plt.xlabel('Time (s)')
            # plt.ylabel('(m)')
            # plt.plot(t, bird.X)
            # plt.plot(t, bird.Y)
            # plt.plot(t, bird.Z)
            # plt.legend(['x', 'y', 'z'])

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.plot(xs = bird.X, ys = bird.Y, zs = bird.Z, zdir = 'z', color = 'orange')
            ax.scatter([v.pos[0] for v in bird.VORTICES],
                        [v.pos[1] for v in bird.VORTICES],
                        [v.pos[2] for v in bird.VORTICES],
                        color = 'pink', s = .1)
            ax.scatter([bird.X[0]], [bird.Y[0]], [bird.Z[0]], 'blue')


            # plt.subplot(212)
            # plt.title('Velocity')
            # plt.xlabel('Time (s)')
            # plt.ylabel('(m/s)')
            # plt.plot(t, bird.U)
            # plt.plot(t, bird.V)
            # plt.plot(t, bird.W)
            # plt.legend(['u', 'v', 'w'])

            plt.show()



    def reset(self, observe = True):
        self.dones = {i: False for i in self.agents}

        self.agent_order = list(self.agents)
        self._agent_selector.reinit(self.agent_order)
        self.agent_selection = self._agent_selector.reset()

        self.positions = self.get_positions()

        self.rewards = {i: 0 for i in self.agents}

    def get_positions(self):
        pos = []
        for agent in self.agents:
            bird = self.birds[agent]
            pos.append([bird.x, bird.y, bird.z])
        return pos
