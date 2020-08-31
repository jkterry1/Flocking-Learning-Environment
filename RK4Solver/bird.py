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
        self.inertia = np.array([[self.Ixx, 0.0, self.Ixz],
                                    [0.0, self.Iyy, 0.0],
                                    [self.Ixz, 0.0, self.Izz]])

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
        self.uvw = np.array([u,v,w])

        '''
        angular velocity:
            p is the rotation about the axis coming out the front of the bird
            q is the rotation about teh axis coming out the right wing of the bird
            r is the rotation about the axis coming out the top of the bird
        '''
        self.p = p
        self.q = q
        self.r = r
        self.pqr = np.array([p, q, r])

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
        self.angles = np.array([phi, theta, psi])

        '''
        position
            x, y, z
        '''
        self.x = x
        self.y = y
        self.z = z
        self.xyz = np.array([x,y,z])

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
        #print()
        #print("Updating uvw")
        uvw = self.take_time_step(de.duvwdt, self.uvw, h)
        u, v, w = uvw
        #print("uvw: ", uvw)
        #print("u: ", u)

        pqr = self.take_time_step(de.dpqrdt, self.pqr, h)
        p, q, r = pqr

        angles = self.take_time_step(de.danglesdt, self.angles, h)
        phi, theta, psi = angles

        xyz = self.take_time_step(de.dxyzdt, self.xyz, h)
        x, y, z = xyz

        self.xyz = xyz
        self.x, self.y, self.z = self.xyz
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

        self.uvw = uvw
        self.u, self.v, self.w = uvw
        self.U.append(u)
        self.V.append(v)
        self.W.append(w)

        self.pqr = pqr
        self.p, self.q, self.r = pqr
        self.P.append(p)
        self.Q.append(q)
        self.R.append(r)

        self.angles = angles
        self.phi, self.theta, self.psi = self.angles
        self.THETA.append(theta)
        self.PHI.append(phi)
        self.PSI.append(psi)

        # Shed a vortex
        # if u > 0 or v > 0 or w > 0:
        #     #right, left
        #     self.vortex_buffer = (Vortex(self, 1), Vortex(self, -1))
        self.vortex_buffer = (Vortex(self, 1), Vortex(self, -1))

    def update_vortex_positions(self, vortices, h):
        for i in range(1, len(vortices)-2):
            #tangent vectors
            t_minus = vortices[i].pos - vortices[i-1].pos
            t = vortices[i+1].pos - vortices[i].pos
            t_plus = vortices[i+2].pos - vortices[i+1].pos

            l_t_minus = np.linalg.norm(t_minus)
            l_t = np.linalg.norm(t)
            #print(np.dot(t_minus, t)/(l_t_minus * l_t))
            if np.dot(t_minus, t)/(l_t_minus * l_t) <= -1.0:
                theta = np.arccos(-1.0)
            if np.dot(t_minus, t)/(l_t_minus * l_t) >= 1.0:
                theta = np.arccos(1.0)
            else:
                theta = np.arccos(np.dot(t_minus, t)/(l_t_minus * l_t))

            #normal vector
            n = (np.cross(t, t_plus)/(1 * np.dot(t, t_plus))) - (np.cross(t_minus, t)/(1 + np.dot(t_minus, t)))
            #binormal vector
            b = np.cross(t, n)

            #strength
            gamma = vortices[i].gamma
            #distance from last point
            epsilon = l_t_minus

            pos = self.take_vortex_time_step(vortices[i].pos, gamma, epsilon, b, theta, h)

            #print(abs(vortices[i].pos - pos))
            vortices[i].dist_travelled += abs(vortices[i].pos - pos)
            vortices[i].pos = pos
            vortices[i].x, vortices[i].y, vortices[i].z = vortices[i].pos

    def shed_vortices(self):
        x = self.vortex_buffer[0].x
        y = self.vortex_buffer[0].y
        z = self.vortex_buffer[0].z
        #print("Shedding vortex at ", [x,y,z])
        self.VORTICES_RIGHT.append(self.vortex_buffer[0])
        self.VORTICES_LEFT.append(self.vortex_buffer[1])

    #returns fu, fv, fw, tu, tv, tw
    def vortex_forces(self, vortices):
        fu, fv, fw = [0.0, 0.0, 0.0]
        tu, tv, tw = [0.0, 0.0, 0.0]
        for vortex in vortices:
            #returns velocities on left and right wing
            L, R = vortex.bird_vel(self)
            # print()
            # print("bird ", self)
            # print("vort ", vortex)
            # print("L ", L)
            # print("R ", R)

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
        for b in range(len(birds)):
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

    def take_vortex_time_step(self, pos, gamma, epsilon, b, theta, h):
        ddt = de.drdt
        k1 = h * ddt(gamma, epsilon, b, theta)
        k2 = h * ddt(gamma, epsilon, b, theta)
        k3 = h * ddt(gamma, epsilon, b, theta)
        k4 = h * ddt(gamma, epsilon, b, theta)

        return pos + (1.0 / 6.0)*(k1 + (2.0 * k2) + (2.0 * k3) + k4)

    def __lt__(self, other):
        return self.x < other.x
