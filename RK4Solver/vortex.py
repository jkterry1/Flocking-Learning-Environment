import numpy as np

class Vortex():

    def __init__(self, bird, sign):
        self.bird = bird
        self.sign = sign
        self.C = 5.0

        '''
        orientation
        '''
        self.phi = bird.phi
        self.theta = bird.theta
        self.psi = bird.psi

        '''
        position
        '''
        mat = self.get_transform(self.phi, self.theta, self.psi)
        a = np.array([0.0, bird.Xl, 0.0])
        if sign == 1: # right wing
            self.pos = np.array([bird.x, bird.y, bird.z]) +  np.matmul(mat, a)
        else: #left wing
            self.pos = np.array([bird.x, bird.y, bird.z]) -  np.matmul(mat, a)
        self.x, self.y, self.z = self.pos

        '''
        properties
        '''
        vel = np.sqrt(bird.u**2 + bird.v**2 + bird.w**2)
        self.gamma = self.C * bird.m/(bird.Xl * vel)
        #print("Gamma: ", self.gamma)
        self.core = 0.05 * bird.Xl
        self.max_r = 1.0

    def earth_vel(self, x, y, z):
        r = np.sqrt((y - self.pos[1])**2 + (z - self.pos[2])**2)
        if r < self.core or r > self.max_r:
            return [0.0, 0.0, 0.0]

        v_tan = self.gamma * r**2 / (2 * np.pi * r * (r**2 + self.core**2))
        r_vec = np.array([0, y - self.y, z - self.z])/r
        tan_vec = v_tan * np.array([0, -r_vec[2], r_vec[1]])

        phi = self.phi
        theta = self.theta
        psi = self.psi
        mat = self.get_transform(phi, theta, psi)

        a = np.matmul(mat, tan_vec)
        return self.sign * a

    def get_transform(self, phi, theta, psi):
        sphi, cphi = np.sin(phi), np.cos(phi)
        stheta, ctheta = np.sin(theta), np.cos(theta)
        spsi, cpsi = np.sin(psi), np.cos(psi)

        mat = [[cphi * ctheta, -spsi * cphi + cpsi * stheta * sphi, spsi * sphi + cpsi * cphi * stheta],
                [spsi * ctheta, cpsi * cphi + sphi * stheta * spsi, -cpsi * sphi + stheta * spsi * cphi],
                [-stheta, ctheta * sphi, ctheta * cphi]]
        mat = np.array(mat)
        return mat
