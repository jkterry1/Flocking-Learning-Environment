import numpy as np
import DiffEqs as de

class Bird():

    def __init__(self,
                u = 0.0, v = 0.0, w = 0.0,
                p = 0.0, q = 0.0, r = 0.0,
                theta = 0.0, phi = 0.0, psi = 0.0,
                vx = 0.0, vy = 0.0, vz = 0.0,
                x = 0.0, y = 0.0, z = 0.0):

        '''
        properties:
            m, mass
        '''
        self.m = 1.0
        self.g = -9.8

        # fixed body frame variables:
        '''
        velocity:
            u points out the front of the bird
            v points out of the right wing
            w points up through the center of the bird
        '''
        self.u = u
        self.v = v
        self.w = w

        '''
        angular velocity:
            p is the rotation about the axis coming out the front of the bird
            q is the rotation about teh axis coming out the right wing of the bird
            r is the rotation about the axis coming out the top of the bird
        '''
        self.p = p
        self.q = q
        self.r = r

        # intertial frame variables
        '''
        euler angles:
            theta is the rotated angle about the intertial x axis
            phi is the rotated angle about the intertial y axis
            psi is the rotated angle about the intertial z axis
        '''
        self.theta = theta
        self.phi = phi
        self.psi = psi

        '''
        velocity:
            vx is the velocity in the intertial x direction
            vy is the velocity in the inertial y direction
            vz is the velocity in the intertial z direction
        '''
        self.vx = vx
        self.vy = vy
        self.vz = vz

        '''
        position
            x, y, z
        '''
        self.x = x
        self.y = y
        self.z = z

        '''
        keep track of old values
        '''
        self.U = [u]
        self.V = [v]
        self.W = [w]

        self.X = [x]
        self.Y = [y]
        self.Z = [z]

    def update(self, t, h):
        # update u, v, w from forces, old angles, and old p,q,r
        u = self.take_time_step(de.dudt, self.u, t, h)
        v = self.take_time_step(de.dvdt, self.v, t, h)
        w = self.take_time_step(de.dwdt, self.w, t, h)

        self.u = u
        self.v = v
        self.w = w
        self.U.append(u)
        self.V.append(v)
        self.W.append(w)

        #update p, q, r from torque

        #calculate new angles from new p,q,r and old angles

        #calculate vx, vy, vz from new u,v,w and new angles
        self.vx, self.vy, self.vz = self.calculate_vel()

        #update position from vx, vy, vz
        x = self.take_time_step(de.dxdt, self.vx, t, h)
        y = self.take_time_step(de.dydt, self.vy, t, h)
        z = self.take_time_step(de.dzdt, self.vz, t, h)

        self.x = x
        self.y = y
        self.z = z
        self.X.append(x)
        self.Y.append(y)
        self.Z.append(z)

        print("u, v, w: ", (self.u, self.v, self.w))
        #print("x, y, z: ", (self.x, self.y, self.z))



    def take_time_step(self, ddt, y, t, h):
        k1 = h * ddt(t, y, self)
        k2 = h * ddt(t + 0.5 * h, y + 0.5 * k1, self)
        k3 = h * ddt(t + 0.5 * h, y + 0.5 * k2, self)
        k4 = h * ddt(t + h, y + k3, self)

        return y + (1.0 / 6.0)*(k1 + (2 * k2) + (2 * k3) + k4)


    def calculate_vel(self):
        vec = np.array([self.u, self.v, self.w]).T
        mat = np.ones((3,3))
        theta = self.theta
        phi = self.phi
        psi = self.psi
        mat[0, 0] = np.cos(theta) * np.cos(psi)
        mat[0, 1] = np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi)
        mat[0, 2] = np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)

        mat[1, 0] = np.cos(theta) * np.sin(psi)
        mat[1, 1] = np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi)
        mat[1, 2] = np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)

        mat[2, 0] = -np.sin(theta)
        mat[2, 1] = np.sin(phi) * np.cos(theta)
        mat[2, 2] = np.cos(phi) * np.cos(theta)

        v = np.matmul(mat, vec)
        #print("v: ", v)
        return v
