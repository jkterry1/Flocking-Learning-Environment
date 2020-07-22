import numpy as np
import DiffEqs as de
from vortex import Vortex
from queue import PriorityQueue
import sys

class Bird():

    def __init__(self,
                u = 0.0, v = 0.0, w = 0.0,
                p = 0.0, q = 0.0, r = 0.0,
                theta = 0.0, phi = 0.0, psi = 0.0,
                alpha_l = 0.0, beta_l = 0.0, alpha_r = 0.0, beta_r = 0.0,
                x = 0.0, y = 0.0, z = 0.0):
        '''
        Earth Properties:
            g, gravity
            rho, air density
        '''
        self.g = -9.8
        self.rho = 1.225

        '''
        bird properties:
            m, mass
            Cl, lift coefficient
            Cd, drag coefficient
            Xl, wing length
            Yl, wing width
            Zl, wing height

        '''
        self.m = 5.0        #goose is ~5kg
        self.Cl_max = 1.6   #experimental value
        self.Cd = 0.3       #experimental value
        self.S = 0.62
        self.Xl = 0.80     #approximate dimensions
        self.Yl = 0.35
        self.Zl = 0.15

        '''
        Moments of Inertia
        '''
        self.Ixx = self.m * self.Xl**2
        self.Iyy = self.m * self.Yl**2
        self.Izz = self.m * self.Zl**2
        self.Ixz = 0.5 * self.m * self.Zl**2    #approximation

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
        wing orientation
        '''
        self.alpha_l = alpha_l
        self.beta_l = beta_l
        self.alpha_r = alpha_r
        self.beta_r = beta_r

        '''
        Observation variables:
            F, net force
            T, net torque
        '''
        self.F = [0.0, 0.0, 0.0]
        self.T = [0.0, 0.0, 0.0]
        '''
        keep track of old values
        '''
        self.U = [u]
        self.V = [v]
        self.W = [w]

        self.X = [x]
        self.Y = [y]
        self.Z = [z]

        self.P = [p]
        self.Q = [q]
        self.R = [r]

        self.THETA = [theta]
        self.PHI = [phi]
        self.PSI = [psi]

        self.ALPHA_L = [alpha_l]
        self.ALPHA_R = [alpha_r]
        self.BETA_L = [beta_l]
        self.BETA_R = [beta_r]

        self.VORTICES_RIGHT = [Vortex(self, 1)]
        self.VORTICES_LEFT = [Vortex(self, -1)]
        self.vortex_force_u, self.vortex_force_v, self.vortex_force_w = [0.0, 0.0, 0.0]
        self.vortex_torque_u, self.vortex_torque_v, self.vortex_torque_w = [0.0, 0.0, 0.0]


    def update(self, thrust, h, vortices):
        self.thrust = thrust

        a = self.vortex_forces(vortices)
        self.vortex_force_u = a[0]
        self.vortex_force_v = a[1]
        self.vortex_force_w = a[2]
        self.vortex_torque_u = a[3]
        self.vortex_torque_v = a[4]
        self.vortex_torque_w = a[5]

        # update u, v, w from forces, old angles, and old p,q,r
        u = self.take_time_step(de.dudt, self.u, h)
        v = self.take_time_step(de.dvdt, self.v, h)
        w = self.take_time_step(de.dwdt, self.w, h)

        #update p, q, r from torque
        p = self.take_time_step(de.dpdt, self.p, h)
        q = self.take_time_step(de.dqdt, self.q, h)
        r = self.take_time_step(de.drdt, self.r, h)

        #calculate new angles from new p,q,r and old angles
        theta = self.take_time_step(de.dthetadt, self.theta, h)
        phi = self.take_time_step(de.dphidt, self.phi, h)
        psi = self.take_time_step(de.dpsidt, self.psi, h)

        #update x,y,z
        x = self.take_time_step(de.dxdt, self.x, h)
        y = self.take_time_step(de.dydt, self.y, h)
        z = self.take_time_step(de.dzdt, self.z, h)

        self.x = x
        self.y = y
        self.z = z
        if self.z <= 0:
            self.z = 0
            u = 0
            v = 0
            w = 0
            p = 0
            q = 0
            r = 0
        self.X.append(x)
        self.Y.append(y)
        self.Z.append(z)

        self.u = u
        self.v = v
        self.w = w
        self.U.append(u)
        self.V.append(v)
        self.W.append(w)

        self.p = p
        self.q = q
        self.r = r
        self.P.append(p)
        self.Q.append(q)
        self.R.append(r)

        self.theta = theta
        self.phi = phi
        self.psi = psi
        self.THETA.append(theta)
        self.PHI.append(phi)
        self.PSI.append(psi)

        # Shed a vortex
        # if u > 0 or v > 0 or w > 0:
        #     #right, left
        #     self.vortex_buffer = (Vortex(self, 1), Vortex(self, -1))
        self.vortex_buffer = (Vortex(self, 1), Vortex(self, -1))

    def shed_vortices(self):
        self.VORTICES_RIGHT.append(self.vortex_buffer[0])
        self.VORTICES_LEFT.append(self.vortex_buffer[1])

    #returns fu, fv, fw, tu, tv, tw
    def vortex_forces(self, vortices):
        fu, fv, fw = [0.0, 0.0, 0.0]
        tu, tv, tw = [0.0, 0.0, 0.0]
        for vortex in vortices:
            #returns velocities on left and right wing
            L, R = vortex.bird_vel(self)
            #print("L ", L)
            #calculate up and down forces
            #left wing
            u, v, w = L
            Aw = self.Xl * self.Yl
            D = np.sign(w) * self.Cd * Aw * (self.rho * w ** 2)/2.0
            fw += D
            tu -= D * self.Xl/2.0

            #right wing
            u, v, w = R
            A = self.Xl * self.Yl
            D = np.sign(w) * self.Cd * A * (self.rho * w ** 2)/2.0
            fw += D
            tu += D * self.Xl/2.0
        return fu, fv, fw, tu, tv, tw

    def seven_nearest(self, birds):
        q = PriorityQueue()
        output = []

        import sys
        for b in birds:
            other = birds[b]
            if other is not self:
                d = np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
                q.put((d, other))

        for i in range(7):
            if not q.empty():
                a, b = q.get()
                output.append(b)

        return output



    def take_time_step(self, ddt, y, h):
        k1 = h * ddt(y, self)
        k2 = h * ddt(y + 0.5 * k1, self)
        k3 = h * ddt(y + 0.5 * k2, self)
        k4 = h * ddt(y + k3, self)

        return y + (1.0 / 6.0)*(k1 + (2.0 * k2) + (2.0 * k3) + k4)

    def __lt__(self, other):
        return self.x < other.x

    '''
    def print_bird(self, bird, action = []):
        file = open('A_errors.txt', 'w')
        file = sys.stderr
        print('-----------------------------------------------------------------------', file = file)
        print("thrust, al, ar, bl, br \n", action, file = file)
        print("x, y, z: \t\t", [bird.x, bird.y, bird.z], file = file)
        print("u, v, w: \t\t", [bird.u, bird.v, bird.w], file = file)
        print("phi, theta, psi: \t", [bird.phi, bird.theta, bird.psi], file = file)
        print("p, q, r: \t\t", [bird.p, bird.q, bird.r], file = file)
        print("alpha, beta (left): \t", [bird.alpha_l, bird.beta_l], file = file)
        print("alpha, beta (right): \t", [bird.alpha_r, bird.beta_r], file = file)
        print("Fu, Fv, Fw: \t\t", bird.F, file = file)
        print("Tu, Tv, Tw: \t\t", bird.T, file = file)
        print("VFu, VFv, VFw: \t\t", bird.vortex_force_u, bird.vortex_force_v, bird.vortex_force_w, file = file)
        print("VTu, VTv, VTw: \t\t", bird.vortex_torque_u, bird.vortex_torque_v, bird.vortex_torque_w, file = file)
        print(file = file)
    '''
