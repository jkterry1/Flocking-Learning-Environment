from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from pettingzoo.utils.env import AECEnv
from supersuit import gym_vec_env
from gym import spaces
import numpy as np
import DiffEqs as de
import bird
from bird import Bird
import matplotlib.pyplot as plt
import copy
import csv
import plotting

def env(**kwargs):
    env = raw_env(**kwargs)
    #env = gym_vec_env(env, 4, multiprocessing=False)
    #env = wrappers.NanNoOpWrapper(env, np.zeros(5), "executing the 'do nothing' action.")
    return env


class raw_env(AECEnv):

    def __init__(self,
                 N = 7,
                 h = 0.001,
                 t = 0.0,
                 birds = None,
                 filename = "episode_1.csv"
                 ):
        self.h = h
        self.t = t
        self.N = N
        self.num_agents = N

        self.act_dims = [5 for i in range(self.N)]

        self.max_r = 1.0
        self.max_steps = 100.0/h

        self.episodes = 0

        self.energy_punishment = 2.0
        self.forward_reward = 5.0
        self.crash_reward = -100.0

        self.agents = ["bird_{}".format(i) for i in range(N)]
        if birds is None:
            birds = [Bird(z = 50.0, y = 3.0*i) for i in range(self.N)]
        self.starting_conditions = [copy.deepcopy(bird) for bird in birds]
        self.birds = {agent: birds[i] for i, agent in enumerate(self.agents)}
        self.agent_order = list(self.agents)
        self._agent_selector = agent_selector(self.agent_order)

        limit = 0.01
        action_space = spaces.Box(low = np.array([0.0, -limit, -limit, -limit, -limit]),
                                        high = np.array([10.0, limit, limit, limit, limit]))
        self.action_spaces = {bird: action_space for bird in self.birds}
        # self.action_space = [action_space for _ in range(self.N)]
        self.action_space = action_space

        low = -10000.0 * np.ones(22 + 6*min(self.N-1, 7),)
        high = 10000.0 * np.ones(22 + 6*min(self.N-1, 7),)
        observation_space = spaces.Box(low = low, high = high)
        self.observation_spaces = {bird: observation_space for bird in self.agents}
        # self.observation_space = [observation_space for _ in range(self.N)]
        self.observation_space = observation_space

        self.data = []
        self.infos = {i:{} for i in range(self.N)}

    def step(self, action, observe = True):
        bird = self.birds[self.agent_selection]
        #print(action)

        if np.isnan(action).any():
            action = np.zeros(5)

        #self.print_bird(bird, action)

        thrust = action[0]

        self.update_angles(action)

        vortices = self.get_vortices(bird)
        #print("vortices: ", vortices)
        bird.update(thrust, self.h, vortices)

        # reward calculation
        reward = 0
        if bird.x > bird.X[-2]:
            reward += self.forward_reward
        reward -= self.energy_punishment * action[0]
        self.rewards[self.agents.index(self.agent_selection)] = reward

        if self.crashed(bird):
            self.dones = {i: True for i in range(self.N)}
            #self.dones = [True for i in range(self.N)]
            print("Done!")
            print("agent ", self.agent_selection, " crashed")
            self.rewards[self.agents.index(self.agent_selection)] = self.crash_reward

        # if we have moved through one complete timestep
        if self.agent_selection == self.agents[-1]:
            self.steps += 1
            for b in self.birds:
                bird = self.birds[b]
                bird.shed_vortices()

                #remove expired vortices
                if self.steps > 1.0/self.h:
                    bird.VORTICES_LEFT.pop(0)
                    bird.VORTICES_RIGHT.pop(0)

            self.positions = self.get_positions()
            self.t = self.t + self.h

        self.agent_selection = self._agent_selector.next()

        if bird.x > 500.0:
            self.dones = {i: True for i in range(self.N)}
            #self.dones = [True for i in range(self.N)]
            #print("Done!")
            #print("agent ", self.agent_selection, " made it to the destination")

        #self.log(bird)

        #print(self.agent_selection)
        #print("z: ", bird.z)

        if observe:
            obs = self.observe()
            #print("dones: ", self.dones)
            return obs, self.rewards, self.dones, self.infos

    def observe(self):
        force = self.birds[self.agent_selection].F
        torque = self.birds[self.agent_selection].T
        bird = self.agent_selection

        bird = self.birds[bird]
        pos += [bird.x, bird.y, bird.z]
        pos += [bird.u, bird.v, bird.w]
        pos += [bird.p, bird.q, bird.r]
        pos += [bird.phi, bird.theta, bird.psi]
        pos += [bird.alpha_l, bird.beta_l, bird.alpha_r, bird.beta_r]
        nearest = bird.seven_nearest(self.birds)
        for other in nearest:
            pos += [other.x - bird.x, other.y - bird.y, other.z - bird.z]
            # print("pos1 ", pos)
            pos += [other.u - bird.u, other.v - bird.v, other.w - bird.w]
            # print("pos2 ", pos)
        obs = np.array(force + torque + pos)
        return obs

    def reset(self, observe = True):

        self.episodes +=1

        self.old_birds = {bird: copy.deepcopy(self.birds[bird]) for bird in self.birds}


        self.agent_order = list(self.agents)
        self._agent_selector.reinit(self.agent_order)
        self.agent_selection = self._agent_selector.reset()

        birds = [copy.deepcopy(bird) for bird in self.starting_conditions]
        self.birds = {agent: birds[i] for i, agent in enumerate(self.agents)}
        self.positions = self.get_positions()

        self.rewards = {i: 0 for i in range(self.N)}

        self.steps = 0

        self.dones = {i:False for i in range(self.N)}
        #self.dones = [False for i in range(self.N)]

        obs = self.observe()

        return obs

    def render(self):
        self.plot_values(show = True)
        self.plot_birds(show = True)

    def update_angles(self, action):
        bird = self.birds[self.agent_selection]
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

    def crashed(self, bird):
        if bird.z <= 0 or bird.z >= 100:
            return True

        lim = 2*np.pi
        if abs(bird.p) > lim or abs(bird.q) > lim or  abs(bird.r) > lim:
            return True

        for b in self.birds:
            other = self.birds[b]
            if other is not bird:
                dist = np.sqrt((bird.x - other.x)**2 + (bird.y - other.y)**2 + (bird.z - other.z)**2)
                return dist < bird.Xl/2.0

    def plot_values(self, show = False):
        plotting.plot_values(self.birds, show)

    def plot_birds(self, plot_vortices = False, show = False):
        plotting.plot_birds(self.birds, plot_vortices, show)

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
