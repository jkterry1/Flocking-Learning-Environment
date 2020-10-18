import numpy as np
from bird import Bird
import copy

class Flock():

    def __init__(self,
                 N = 10,
                 h = 0.001,
                 t = 0.0,
                 bird_starts=None,
                 LIA = False
                 ):
        self.h = h
        self.t = t
        self.N = N
        self.num_agents = N
        self.LIA = LIA

        self.total_vortices = 0.0
        self.total_dist = 0.0

        self.max_r = 1.0
        self.max_steps = 100.0/self.h

        self.energy_punishment = 2.0
        self.forward_reward = 5.0
        self.crash_reward = -100.0

        self.agents = range(self.N)
        birds = None
        if bird_starts is None:
            bird_starts = [dict(z = 50.0, y = 3.0*i, u=5.0) for i in range(self.N)]

        birds = [Bird(**start_val) for start_val in bird_starts]
        self.starting_conditions = [copy.deepcopy(bird) for bird in birds]
        self.birds = birds

    def reset(self):
        self.old_birds = [copy.deepcopy(self.birds[i]) for i in range(self.N)]
        birds = [copy.deepcopy(bird) for bird in self.starting_conditions]
        self.birds = birds

    def update_bird(self, action, agent):
        thrust = action[0]
        bird = self.birds[agent]
        self.update_angles(action, agent)
        vortices = self.get_vortices(bird)
        bird.update(thrust, self.h, vortices)

    def get_reward(self, action, agent):
        reward = 0
        done = False
        bird = self.birds[agent]
        if bird.x > bird.X[-2]:
            reward += self.forward_reward
        reward -= self.energy_punishment * action[0]

        if self.crashed(bird):
            done = True
            reward = self.crash_reward

        if bird.x > 500.0:
            done = True

        return done, reward

    def crashed(self, bird):
        if bird.z <= 0 or bird.z >= 100:
            return True

        lim = 2*np.pi
        if abs(bird.p) > lim or abs(bird.q) > lim or  abs(bird.r) > lim:
            return True

        crash = False
        for b in range(len(self.birds)):
            other = self.birds[b]
            if other is not bird:
                dist = np.sqrt((bird.x - other.x)**2 + (bird.y - other.y)**2 + (bird.z - other.z)**2)
                if dist < bird.Xl/2.0:
                    crash = True
        return crash

    def get_vortices(self, curr):
        vortices = []
        for b in range(len(self.birds)):
            bird = self.birds[b]
            if bird is not curr:
                for vorts in [bird.VORTICES_LEFT, bird.VORTICES_RIGHT]:
                    i = 0
                    v = vorts[i]

                    #want first vortex ahead of current vortex
                    while i < len(vorts) and v.x < curr.x:
                        v = vorts[i]
                        i = i+1
                    if i < len(vorts) and v.x >= curr.x:
                        r = np.sqrt((curr.y - v.y)**2 + (curr.z - v.z)**2)
                        if r < self.max_r:
                            vortices.append(v)
        return vortices

    def get_observation(self, agent):
        force = self.birds[agent].F
        torque = self.birds[agent].T

        bird = self.birds[agent]
        pos = []
        pos += [bird.x, bird.y, bird.z]
        pos += [bird.u, bird.v, bird.w]
        pos += [bird.p, bird.q, bird.r]
        pos += [bird.phi, bird.theta, bird.psi]
        pos += [bird.alpha_l, bird.beta_l, bird.alpha_r, bird.beta_r]
        nearest = bird.seven_nearest(self.birds)
        for other in nearest:
            pos += [other.x - bird.x, other.y - bird.y, other.z - bird.z]
            pos += [other.u - bird.u, other.v - bird.v, other.w - bird.w]

        obs = np.array(force + torque + pos)
        return obs

    def update_vortices(self):
        for b in range(len(self.birds)):
            bird = self.birds[b]
            bird.shed_vortices()

            if self.LIA:
                bird.update_vortex_positions(bird.VORTICES_RIGHT, self.h*10.0)
                bird.update_vortex_positions(bird.VORTICES_LEFT, self.h*10.0)

            #remove expired vortices
            if len(bird.VORTICES_LEFT) > 1.0/self.h:
                a = bird.VORTICES_LEFT.pop(0)
                b = bird.VORTICES_RIGHT.pop(0)

    def update_angles(self, action, agent):
        bird = self.birds[agent]
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


    def get_birds(self):
        return self.birds


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
