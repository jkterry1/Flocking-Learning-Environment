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
import copy

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
                 h = 0.01,
                 t = 0.0,
                 birds = None
                 ):
        self.h = h
        self.t = t
        self.N = N

        self.max_r = 1.0
        self.max_steps = 1000000

        self.agents = ["bird_{}".format(i) for i in range(N)]
        if birds is None:
            birds = [Bird(z = 100.0) for agent in self.agents]
        self.starting_conditions = [copy.deepcopy(bird) for bird in birds]
        self.birds = {agent: birds[i] for i, agent in enumerate(self.agents)}
        self.agent_order = list(self.agents)
        self._agent_selector = agent_selector(self.agent_order)

        self.action_spaces = {bird: spaces.Box(low = 0.0* np.zeros(4), high = 5.0 * np.ones(4)) for bird in self.birds}
        #self.action_space = spaces.Box(np.array([0.0, -5.0, -5.0, -5.0]), high = 1000.0 * np.ones(4))
        #self.action_space = self.observation_space = spaces.Box(low=0.0, high=50.0,
                                        #shape=(1,), dtype = np.float32)
        self.observation_space = spaces.Box(low = -1000.0*np.ones(9), high = 1000.0 * np.ones(9))
        self.metadata = []

    def step(self, action, observe = True):
        bird = self.birds[self.agent_selection]

        self.print_bird(bird, action)

        thrust = action[0]
        #torque = [0.0, 0.0, 0.0]
        #print("thrust ", thrust)
        limit = np.pi/4.0
        bird.alpha_l += action[1]
        if bird.alpha_l > limit:
            bird.alpha_l = limit
        if bird.alpha_l < -limit:
            bird.alpha_l = -limit

        bird.beta_l += action[2]
        if bird.beta_l > limit:
            bird.beta_l = limit
        if bird.beta_l < -limit:
            bird.beta_l = -limit

        bird.alpha_r += action[3]
        if bird.alpha_r > limit:
            bird.alpha_r = limit
        if bird.alpha_r < -limit:
            bird.alpha_r = -limit

        bird.beta_r += action[4]
        if bird.beta_r > limit:
            bird.beta_r = limit
        if bird.beta_r < -limit:
            bird.beta_r = -limit

        vortices = self.get_vortices(bird)
        print("vortices: ", vortices)
        bird.update(thrust, self.h, vortices)


        if bird.z <= 0 or bird.z > 200:
            self.reward = -100.0
            self.done = True

        # if we have moved through one complete timestep
        if self.agent_selection == self.agents[-1]:
            for b in self.birds:
                bird = self.birds[b]
                bird.shed_vortices()
            self.positions = self.get_positions()
            self.t = self.t + self.h

        self.agent_selection = self._agent_selector.next()
        #self.info['episode'] += 1

        self.steps += 1
        if self.steps > self.max_steps:
            self.done = True

        obs = self.observe()
        for i in range(len(obs)):
            if obs[i] > 1000:
                self.rewards[self.agent_selection] = -100.0
                #self.done = True
                #print("{} was large, {}", i, obs[i])
        if observe:
            return obs, self.reward, self.done, self.info

    def observe(self):
        force = self.birds[self.agent_selection].F
        torque = self.birds[self.agent_selection].T
        bird = self.agent_selection
        pos = [self.birds[bird].x, self.birds[bird].y, self.birds[bird].z]
        obs = force+torque + pos
        return obs

    def plot_values(self):
        plt1 = plt.subplot(311)
        plt2 = plt.subplot(312)
        plt3 = plt.subplot(313)
        for bird in self.birds:
            bird = self.birds[bird]
            t = np.arange(len(bird.U))

            plt1.title.set_text('position')
            plt1.set_xlabel('Time (s)')
            plt1.set_ylabel('m')
            plt1.plot(t, bird.X)
            plt1.plot(t, bird.Y)
            plt1.plot(t, bird.Z)
            leg = []
            for _ in self.birds:
                leg += ['x', 'y', 'z']
            plt1.legend(leg)

            plt2.title.set_text('angular vel')
            plt2.set_xlabel('Time (s)')
            plt2.set_ylabel('rad/s')
            plt2.plot(t, bird.P)
            plt2.plot(t, bird.Q)
            plt2.plot(t, bird.R)
            leg = []
            for _ in self.birds:
                leg += ['p', 'q', 'r']
            plt2.legend(leg)

            plt3.title.set_text('velocity')
            plt3.set_xlabel('Time (s)')
            plt3.set_ylabel('(m/s)')
            plt3.plot(t, bird.U)
            plt3.plot(t, bird.V)
            plt3.plot(t, bird.W)
            leg = []
            for _ in self.birds:
                leg += ['u', 'v', 'w']
            plt3.legend(leg)
        plt.show()

    def plot_birds(self, plot_vortices = False):
        first = plot_vortices
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # ax.set_xlim3d(0,5)
        # ax.set_ylim3d(-2.5,2.5)
        # ax.set_zlim3d(0,5)
        for b in self.birds:
            bird = self.birds[b]
            ax.plot(xs = bird.X, ys = bird.Y, zs = bird.Z, zdir = 'z', color = 'orange')
            ax.scatter([v.pos[0] for v in bird.VORTICES_LEFT],
                        [v.pos[1] for v in bird.VORTICES_LEFT],
                        [v.pos[2] for v in bird.VORTICES_LEFT],
                        color = 'red', s = .5)
            ax.scatter([v.pos[0] for v in bird.VORTICES_RIGHT],
                        [v.pos[1] for v in bird.VORTICES_RIGHT],
                        [v.pos[2] for v in bird.VORTICES_RIGHT],
                        color = 'red', s = .5)
            ax.scatter([bird.X[0]], [bird.Y[0]], [bird.Z[0]], 'blue')

        x = []; y = []; z = []
        u = []; v = []; w = []
        r = 0.25
        if first:
            first = False
            for vort in bird.VORTICES_LEFT:
                x.append(vort.x);y.append(vort.y + r);z.append(vort.z + r)
                a,b,c = vort.earth_vel(vort.x, vort.y + r, vort.z + r)
                u.append(a); v.append(b); w.append(c)

                x.append(vort.x);y.append(vort.y - r);z.append(vort.z - r)
                a,b,c = vort.earth_vel(vort.x, vort.y - r, vort.z - r)
                u.append(a);v.append(b);w.append(c)

                x.append(vort.x);y.append(vort.y);z.append(vort.z + r)
                a,b,c = vort.earth_vel(vort.x, vort.y, vort.z + r)
                u.append(a); v.append(b); w.append(c)

                x.append(vort.x);y.append(vort.y);z.append(vort.z - r)
                a,b,c = vort.earth_vel(vort.x, vort.y, vort.z - r)
                u.append(a); v.append(b); w.append(c)

                x.append(vort.x);y.append(vort.y - r);z.append(vort.z + r)
                a,b,c = vort.earth_vel(vort.x, vort.y - r, vort.z + r)
                u.append(a); v.append(b); w.append(c)

                x.append(vort.x);y.append(vort.y + r);z.append(vort.z - r)
                a,b,c = vort.earth_vel(vort.x, vort.y + r, vort.z - r)
                u.append(a);v.append(b);w.append(c)
            for vort in bird.VORTICES_RIGHT:
                x.append(vort.x);y.append(vort.y + r);z.append(vort.z + r)
                a,b,c = vort.earth_vel(vort.x, vort.y + r, vort.z + r)
                u.append(a); v.append(b); w.append(c)

                x.append(vort.x);y.append(vort.y - r);z.append(vort.z - r)
                a,b,c = vort.earth_vel(vort.x, vort.y - r, vort.z - r)
                u.append(a);v.append(b);w.append(c)

                x.append(vort.x);y.append(vort.y);z.append(vort.z + r)
                a,b,c = vort.earth_vel(vort.x, vort.y, vort.z + r)
                u.append(a); v.append(b); w.append(c)

                x.append(vort.x);y.append(vort.y);z.append(vort.z - r)
                a,b,c = vort.earth_vel(vort.x, vort.y, vort.z - r)
                u.append(a); v.append(b); w.append(c)

                x.append(vort.x);y.append(vort.y - r);z.append(vort.z + r)
                a,b,c = vort.earth_vel(vort.x, vort.y - r, vort.z + r)
                u.append(a); v.append(b); w.append(c)

                x.append(vort.x);y.append(vort.y + r);z.append(vort.z - r)
                a,b,c = vort.earth_vel(vort.x, vort.y + r, vort.z - r)
                u.append(a);v.append(b);w.append(c)


        ax.quiver(x,y,z,u,v,w, length = .1, normalize = True)
        plt.show()


    def reset(self, observe = True):
        self.dones = {i: False for i in self.agents}

        self.agent_order = list(self.agents)
        self._agent_selector.reinit(self.agent_order)
        self.agent_selection = self._agent_selector.reset()

        birds = [copy.deepcopy(bird) for bird in self.starting_conditions]
        #print("bird ", birds[0].z)
        self.birds = {agent: birds[i] for i, agent in enumerate(self.agents)}
        self.positions = self.get_positions()

        self.rewards = {i: 0 for i in self.agents}

        self.info = {'episode': None}

        self.steps = 0
        self.reward = 0.0

        self.done = False

        return self.observe()

    def get_positions(self):
        pos = []
        for agent in self.agents:
            bird = self.birds[agent]
            pos.append([bird.x, bird.y, bird.z])
        return pos

    def get_vortices(self, curr):
        vortices = []
        for b in self.birds:
            bird = self.birds[b]
            if bird is not curr:
                for vorts in [bird.VORTICES_LEFT, bird.VORTICES_RIGHT]:
                    i = 0
                    v = vorts[i]
                    #want first vortex ahead of it
                    while i < len(vorts) and v.x < curr.x:
                        v = vorts[i]
                        i = i+1
                    if v.x >= curr.x:
                        r = np.sqrt((curr.y - v.y)**2 + (curr.z - v.z)**2)
                        if i < len(vorts) and r < self.max_r:
                            vortices.append(v)
        return vortices

    def print_bird(self, bird, action):
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
