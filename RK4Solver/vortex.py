import numpy as np

class Vortex():

    def __init__(self, bird, sign):
        self.bird = bird
        self.sign = sign
        self.C = 25.0
        self.min_vel = 1.0

        '''
        orientation
        '''
        self.phi = bird.phi
        self.theta = bird.theta
        self.psi = bird.psi

        '''
        position
        '''
        if sign == 1: # right wing
            self.phi += bird.beta_r
        else:
            self.phi -= bird.beta_l
        mat = self.get_transform(self.phi, self.theta, self.psi)
        a = np.array([0.0, bird.Xl, 0.0])
        if sign == 1: # right wing
            self.pos = np.array([bird.x, bird.y, bird.z]) -  np.matmul(mat, a)
        else: #left wing
            self.pos = np.array([bird.x, bird.y, bird.z]) +  np.matmul(mat, a)
        self.x, self.y, self.z = self.pos

        '''
        properties
        '''
        vel = np.sqrt(bird.u**2 + bird.v**2 + bird.w**2)
        if abs(vel) > self.min_vel:
            self.gamma = self.C * bird.m/(bird.Xl * vel)
        else:
            self.gamma = 0.0
        #print("Gamma: ", self.gamma)
        self.core = 0.05 * bird.Xl
        self.max_r = 0.5

    # vlocity of the vortex in the earth's frame
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

    # velocity in the bird's frame
    def bird_vel(self, bird):
        mat = self.get_transform(bird.phi, bird.theta, bird.psi)
        add = np.array([0.0, bird.Xl/2.0, 0.0])

        pos_right = np.array([bird.x, bird.y, bird.z]) +  np.matmul(mat, add)
        pos_left = np.array([bird.x, bird.y, bird.z]) -  np.matmul(mat, add)

        r_right = np.sqrt((pos_right[1] - self.pos[1])**2 + (pos_right[2] - self.pos[2])**2)
        r_left = np.sqrt((pos_left[1] - self.pos[1])**2 + (pos_left[2] - self.pos[2])**2)

        v_tan_l = self.gamma * r_left**2 / (2 * np.pi * r_left * (r_left**2 + self.core**2))
        v_tan_r = self.gamma * r_right**2 / (2 * np.pi * r_right * (r_right**2 + self.core**2))

        r_vec_right = np.array([0, pos_right[1] - self.y, pos_right[2] - self.z])/r_right
        r_vec_left = np.array([0, pos_left[1] - self.y, pos_left[2] - self.z])/r_left

        tan_vec_left = v_tan_l * np.array([0, -r_vec_left[2], r_vec_left[1]])
        tan_vec_right = v_tan_r * np.array([0, -r_vec_right[2], r_vec_right[1]])

        phi = bird.phi - self.phi
        theta = bird.theta - self.theta
        psi = bird.psi - self.psi
        mat = self.get_transform(phi, theta, psi)

        left = np.matmul(mat, tan_vec_left)
        right = np.matmul(mat, tan_vec_right)

        if r_left < self.core or r_left > self.max_r:
            left = np.array([0.0, 0.0, 0.0])
        if r_right < self.core or r_right > self.max_r:
            right = np.array([0.0, 0.0, 0.0])

        ret = [self.sign * left, self.sign * right]
        return ret

    def get_transform(self, phi, theta, psi):
        sphi, cphi = np.sin(phi), np.cos(phi)
        stheta, ctheta = np.sin(theta), np.cos(theta)
        spsi, cpsi = np.sin(psi), np.cos(psi)

        mat = [[cphi * ctheta, -spsi * cphi + cpsi * stheta * sphi, spsi * sphi + cpsi * cphi * stheta],
                [spsi * ctheta, cpsi * cphi + sphi * stheta * spsi, -cpsi * sphi + stheta * spsi * cphi],
                [-stheta, ctheta * sphi, ctheta * cphi]]
        mat = np.array(mat)
        return mat
