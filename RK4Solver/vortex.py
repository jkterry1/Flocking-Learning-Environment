import numpy as np

class Vortex():

    def __init__(self, bird, sign):
        self.bird = bird
        self.sign = sign

        '''
        orientation
        '''
        self.phi = bird.phi
        self.theta = bird.theta
        self.psi = bird.psi

        '''
        position
        '''
        #-------------make sure position accounts for being at the tip of the wing!!!!!!!
        mat = self.get_transform(bird)
        if sign == 1: # right wing
            a = np.array([0.0, bird.Xl, 0.0])
            self.pos = np.array([bird.x, bird.y, bird.z]) +  np.matmul(mat, a)
        else: #left wing
            a = np.array([0.0, bird.Xl, 0.0])
            self.pos = np.array([bird.x, bird.y, bird.z]) -  np.matmul(mat, a)

        '''
        properties
        '''
        self.C = 1.0
        vel = np.sqrt(bird.u**2 + bird.v**2 + bird.w**2)
        self.gamma = bird.m/(bird.Xl * vel)
        #print("Gamma: ", self.gamma)
        self.core = 0.05 * bird.Xl
        self.max_r = 1.0


    #returns velocity of vortex in the bird's frame
    def vel(self, bird):
        r = np.sqrt((bird.x - self.x)**2 + (bird.y - self.y)**2 + (bird.z - self.z)**2)
        if r < self.core or r > self.max_r:
            return [0.0, 0.0, 0.0]
        v_tan = self.gamma * r**2 / (2 * np.pi * r * (r**2 + self.core**2))
        u_v = 0.0
        v_v = self.sign * v_tan * bird.z / r
        w_v = -self.sign * v_tan * bird.y / r
        vec = np.array([u_v, v_v, w_v])

        mat = self.get_transform(bird)

        a = np.matmul(mat, vec)
        (ub, vb, wb) = a
        return a

    def get_transform(self, bird):
        phi = self.phi - bird.phi
        theta = self.theta - bird.theta
        psi = self.psi - bird.psi

        sphi, cphi = np.sin(phi), np.cos(phi)
        stheta, ctheta = np.sin(theta), np.cos(theta)
        spsi, cpsi = np.sin(psi), np.cos(psi)

        mat = [[cphi * ctheta, -spsi * cphi + cpsi * stheta * sphi, spsi * sphi + cpsi * cphi * stheta],
                [spsi * ctheta, cpsi * cphi + sphi * stheta * spsi, -cpsi * sphi + stheta * spsi * cphi],
                [-stheta, ctheta * sphi, ctheta * cphi]]
        mat = np.array(mat)
        return mat
