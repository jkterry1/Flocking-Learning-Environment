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
import csv

def env(**kwargs):
    env = raw_env(**kwargs)
    return env


class raw_env(AECEnv):

    def __init__(self,
                 N = 1,
                 h = 0.001,
                 t = 0.0,
                 birds = None
                 ):
        self.h = h
        self.t = t
        self.N = N
        self.num_agents = N

        self.max_r = 1.0
        self.max_steps = 100.0/h

        self.episodes = 0

        self.agents = ["bird_{}".format(i) for i in range(N)]
        if birds is None:
            birds = [Bird(z = 50.0) for agent in self.agents]
        self.starting_conditions = [copy.deepcopy(bird) for bird in birds]
        self.birds = {agent: birds[i] for i, agent in enumerate(self.agents)}
        self.agent_order = list(self.agents)
        self._agent_selector = agent_selector(self.agent_order)

        self.action_spaces = {bird: spaces.Box(low = 0.0* np.zeros(4), high = 5.0 * np.ones(4)) for bird in self.birds}
        limit = 0.01
        self.action_space = spaces.Box(low = np.array([0.0, -limit, -limit, -limit, -limit]),
                                        high = np.array([10.0, limit, limit, limit, limit]))

        low = -10000.0 * np.ones(9,)
        high = 10000.0 * np.ones(9,)
        self.observation_space = spaces.Box(low = low, high = high)

        self.observation_spaces = {bird: self.observation_space for bird in self.agents}
        self.metadata = []

        self.reward_range = [0, 1.0]

        self.file = open('episode_2.csv', 'a+', newline ='')
        self.data = []

    def step(self, action, observe = True):
        bird = self.birds[self.agent_selection]

        #self.print_bird(bird, action)

        thrust = action[0]

        limit_alpha = np.pi/6.0
        limit_beta = np.pi/4.0
        new_al = bird.alpha_l + action[1]
        if new_al > limit_alpha:
            new_al = limit_alpha
        if new_al < -limit_alpha:
            new_al = -limit_alpha
        bird.alpha_l = new_al

        new_bl = bird.beta_l + action[2]
        if new_bl > limit_beta:
            new_bl = limit_beta
        if new_bl < -limit_beta:
            new_bl = -limit_beta
        bird.beta_l = new_bl

        new_ar = bird.alpha_r + action[3]
        if new_ar > limit_alpha:
            new_ar = limit_alpha
        if new_ar < -limit_alpha:
            new_ar = -limit_alpha
        bird.alpha_r = new_ar

        new_br = bird.beta_r + action[4]
        if new_br > limit_beta:
            new_br = limit_beta
        if new_br < -limit_beta:
            new_br = -limit_beta
        bird.beta_r = new_br

        vortices = self.get_vortices(bird)
        #print("vortices: ", vortices)
        self.done = bird.update(thrust, self.h, vortices)

        self.reward = 1.0

        if self.crashed(bird):
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
        if bird.x > 100.0:
            self.done = True

        obs = self.observe()

        self.log(bird)

        if self.done:
            with self.file:
                write = csv.writer(self.file)
                write.writerows(self.data)

        if observe:
            return obs, self.reward, self.done, self.info

    def observe(self):
        force = self.birds[self.agent_selection].F
        torque = self.birds[self.agent_selection].T
        bird = self.agent_selection
        pos = [self.birds[bird].x, self.birds[bird].y, self.birds[bird].z]
        obs = np.array(force + torque + pos)
        return obs

    def render(self, mode, **kwargs):
        plt.show()

    def crashed(self, bird):
        if bird.z <= 0 or bird.z >= 100:
            return True

        lim = 2*np.pi
        if abs(bird.p) > lim or abs(bird.q) > lim or  abs(bird.r) > lim:
            return True

    def plot_values(self, show = False):
        fig = plt.figure(0)
        plt.clf()
        plt1 = fig.add_subplot(411)
        plt2 = fig.add_subplot(412)
        plt3 = fig.add_subplot(413)
        plt4 = fig.add_subplot(414)
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

            plt3.title.set_text('angle')
            plt3.set_xlabel('Time (s)')
            plt3.set_ylabel('rad')
            plt3.plot(t, bird.PHI)
            plt3.plot(t, bird.THETA)
            plt3.plot(t, bird.PSI)
            leg = []
            for _ in self.birds:
                leg += ['phi', 'theta', 'psi']
            plt3.legend(leg)

            plt4.title.set_text('velocity')
            plt4.set_xlabel('Time (s)')
            plt4.set_ylabel('(m/s)')
            plt4.plot(t, bird.U)
            plt4.plot(t, bird.V)
            plt4.plot(t, bird.W)
            leg = []
            for _ in self.birds:
                leg += ['u', 'v', 'w']
            plt4.legend(leg)

            if show:
                plt.show()

    def plot_birds(self, plot_vortices = False, show = False):
        first = plot_vortices
        fig = plt.figure(1)
        plt.clf()
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
        if show:
            plt.show()

    def reset(self, observe = True):
        plt.figure(0)
        plt.clf()
        plt.figure(1)
        plt.clf()
        self.plot_birds()
        self.plot_values()

        if self.episodes % 10 == 0:
            plt.show()

        self.episodes +=1

        self.old_birds = {bird: copy.deepcopy(self.birds[bird]) for bird in self.birds}

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
