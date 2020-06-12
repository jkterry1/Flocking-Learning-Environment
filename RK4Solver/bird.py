import numpy as np
import DiffEqs as de

class Bird():

    def __init__(self,
                u = 0.0, v = 0.0, w = 0.0,
                p = 0.0, q = 0.0, r = 0.0,
                theta = 0.0, phi = 0.0, psi = 0.0,
                x = 0.0, y = 0.0, z = 0.0):

        '''
        properties:
            m, mass
            g, gravity
            Cl, lift coefficient
            Cd, drag coefficient
            rho, air density
            Au, area looking down u axis
            Av, area looking down v axis
            Aw, area looking down w axis
        '''
        self.m = 1.0
        self.g = -9.8
        self.Cl = 1.2
        self.Cd = 0.3
        self.S = 0.62
        self.rho = 1.225
        self.Au = 0.1
        self.Av = 0.2
        self.Aw = 0.6

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
        u = self.take_time_step(de.dudt, self.u, h)
        v = self.take_time_step(de.dvdt, self.v, h)
        w = self.take_time_step(de.dwdt, self.w, h)

        self.u = u
        self.v = v
        self.w = w
        self.U.append(u)
        self.V.append(v)
        self.W.append(w)

        #update p, q, r from torque

        #calculate new angles from new p,q,r and old angles

        #update x,y,z
        x = self.take_time_step(de.dxdt, self.x, h)
        y = self.take_time_step(de.dydt, self.y, h)
        z = self.take_time_step(de.dzdt, self.z, h)

        self.x = x
        self.y = y
        self.z = z
        self.X.append(x)
        self.Y.append(y)
        self.Z.append(z)

        print("u, v, w: ", (self.u, self.v, self.w))
        print("x, y, z: ", (self.x, self.y, self.z))



    def take_time_step(self, ddt, y, h):
        k1 = h * ddt(y, self)
        k2 = h * ddt(y + 0.5 * k1, self)
        k3 = h * ddt(y + 0.5 * k2, self)
        k4 = h * ddt(y + k3, self)

        return y + (1.0 / 6.0)*(k1 + (2.0 * k2) + (2.0 * k3) + k4)
