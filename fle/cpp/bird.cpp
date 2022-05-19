#include <array>
#include <vector>
#include <queue>
#include <cassert>
#include "vortex.hpp"
#include "utils.hpp"
#include "DiffEqs.hpp"
#include <iostream>
#include <cmath>

using PrevValues = std::vector<double>;
using PrevVectors = std::vector<Vector3d>;
using VortexForces = std::array<Vector3d,2>;
using Vorticies = std::vector<Vortex>;
typedef Vector3d (DiffEqType)(Vector3d abc, Bird & bird);

using namespace std;

struct BirdInit{
    //position along x, y, and z axes
    Vector3d xyz;

    //velocities in the u (forward), v (out from the right wing), and w (up) directions
    Vector3d uvw;

    //angle bird is rotated around the inertial x, y, and z axes
    Vector3d ang;

    //angular velocities around the u, v, and w axes
    Vector3d pqr;

    BirdInit(
            double x, double y, double z,
            double u, double v, double w,
            double theta, double phi, double psi,
            double p, double q, double r
            ){
        BirdInit & self = *this;
        self.xyz = Vector3d(x, y, z);
        self.uvw = Vector3d(u, v, w);
        self.ang = Vector3d(theta, phi, psi);
        self.pqr = Vector3d(p, q, r);
    }
};
using BirdInits = std::vector<BirdInit>;
struct Bird{
    using Birds = std::vector<Bird>;
    /*
    Earth Properties:
        g, gravity
        rho, air density
    */
    static constexpr double g = -9.8;
    static constexpr double rho = 1.225;

    /*
    bird properties:
        m, mass
        Cl_max, maximum lift coefficient, this can change with angle of atack
        Cd, drag coefficient
        Xl, wing length
        Yl, wing width
        Zl, wing height
    */
    static constexpr double m = 5.0;        //goose is ~5kg
    static constexpr double Cl_max = 1.6;   //experimental value
    static constexpr double Cd = 0.3;       //experimental value
    static constexpr double S = 0.62;
    static constexpr double Xl = 0.80;     //approximate dimensions
    static constexpr double Yl = 0.35;
    static constexpr double Zl = 0.8;

    double pi = 2.0 * acos(0.0);

    /*
      Broken is used to keep track of whether the bird has reached extreme
      conditions where the simulation can no longer keep up.
      For example, if the bird is spinning at a rate faster than
      the update rate, the simulation breaks down.
    */
    bool broken;

    /*
    Moments of Inertia
    */
    double Ixx;
    double Iyy;
    double Izz;
    double Ixz;
    Matrix3d inertia;

    /*
    position
        x, y, z
    */
    Vector3d xyz;

    // fixed body frame variables:
    /*
    velocity:
        u points out the front of the bird
        v points out of the right wing
        w points up through the center of the bird
    */
    Vector3d uvw;

    /*
    angular velocity:
        p is the rotation about the axis coming out the front of the bird
        q is the rotation about teh axis coming out the right wing of the bird
        r is the rotation about the axis coming out the top of the bird
    */
    Vector3d pqr;

    // intertial frame variables
    /*
    euler angles:
        theta is the rotated angle about the intertial x axis
        phi is the rotated angle about the intertial y axis
        psi is the rotated angle about the intertial z axis
    */
    Vector3d ang;

    //velocity in earth frame
    Vector3d vxyz;

    /*
      wing orientation,
      alpha is how many degrees the wing is tilting forward or back
      beta is how many degrees the wint is lifted up or down
      More explanation is in the diagrams
    */
    double alpha_l;
    double beta_l;
    double alpha_r;
    double beta_r;

    /*
    Observation variables:
        F, net force
        T, net torque
    */
    Vector3d F;
    //Vector3d Fd;
    Vector3d T;

    /*
      Vortex buffers keep track of the vortices that will
      be generated during this time step
    */
    Vortex vortex_buffer_left;
    Vortex vortex_buffer_right;

    /*
      These store all of the previous locations of the vortex line
      points for logging purposes
    */
    Vorticies VORTICES_RIGHT;
    Vorticies VORTICES_LEFT;

    /*
      The forces  and torques due to vortices (produced by other birds)
      on the current bird in each of the 3 main directions.
    */
    Vector3d vortex_fuvw = Vector3d(0, 0, 0);
    Vector3d vortex_tuvw = Vector3d(0, 0, 0);

    //The bird's thrust, which comes entirely from the bird's chosen action
    double thrust;

    /*
    keep track of old values
    */
    PrevVectors prev_xyz, prev_uvw, prev_ang, prev_pqr;

    PrevValues ALPHA_L;
    PrevValues ALPHA_R;
    PrevValues BETA_L;
    PrevValues BETA_R;

    double max_dist = 0.0;

    Bird() = default;
    Bird(const BirdInit & init){
        Bird & self = *this;
        self.xyz = init.xyz;
        self.uvw = init.uvw;
        self.ang = init.ang;
        self.pqr = init.pqr;

        double alpha_l = 0.0, beta_l = 0.0, alpha_r = 0.0, beta_r = 0.0;

        /*
        Moments of Inertia
        */
        self.Ixx = self.m * sqr(self.Xl);
        self.Iyy = self.m * sqr(self.Yl);
        self.Izz = self.m * sqr(self.Zl);
        self.Ixz = 0.5 * self.m * sqr(self.Zl);    //approximation
        //self.Ixz = 0.0;
        self.inertia = matrix(self.Ixx, 0.0, self.Ixz,
                                    0.0, self.Iyy, 0.0,
                                    self.Ixz, 0.0, self.Izz);


        //Total force and torque on this bird
        self.F = Vector3d(0.0, 0.0, 0.0);
      	self.T = Vector3d(0.0, 0.0, 0.0);

        //Drag Force for use in reward calculation
        //self.Fd = Vector3d(0.0, 0.0, 0.0);

        /*
        wing orientation angles alpha and beta for the left and right wings,
        refer to diagrams for more explanation
        */
        self.alpha_l = alpha_l;
        self.beta_l = beta_l;
        self.alpha_r = alpha_r;
        self.beta_r = beta_r;

        /*
        keep track of old values
        */
        self.VORTICES_RIGHT = Vorticies{Vortex(self, 1)};
        self.VORTICES_LEFT = Vorticies{Vortex(self, -1)};
        self.update_history(self);
    }

    /*
      updates all values (position, orientation, velocity, angular velocity)
      for this bird for one time step given the thrust and relevant vorticies.
    */
    void update(double thrust, double h, Vorticies & vortices){
        Bird & self = *this;
        self.thrust = thrust;

        /*
          Get the values of force and torque due to any vortices the
          bird is interacting with
        */
        VortexForces vf = self.vortex_forces(vortices);
        self.vortex_fuvw = vf[0];
        self.vortex_tuvw = vf[1];

        //Calculate and update velocities for the next time step using the diffeq solver
        Vector3d uvw = self.take_time_step(duvwdt, self.uvw, h);
        //Calculate and update angular velocities for the next time step
        Vector3d pqr = self.take_time_step(dpqrdt, self.pqr, h);
        //calculate and update position for the next time step
        Vector3d xyz = self.take_time_step(dxyzdt, self.xyz, h);
        //calculate and update orientation for the next time step
        Vector3d angles = self.take_time_step(danglesdt, self.ang, h);
        angles[0] = fmod(angles[0], (2.0*self.pi));
        angles[1] = fmod(angles[1], (2.0*self.pi));
        angles[2] = fmod(angles[2], (2.0*self.pi));

        self.xyz = xyz;
        self.uvw = uvw;
        self.ang = angles;
        self.pqr = pqr;

        /*
          Check if the bird has crashed into the ground.
          If so, set vertical position, all velocities,
          and angular velocities to 0 (bird no longer moving)
        */
        if(self.xyz[2] <= 0){
            self.xyz = Vector3d(0, 0, 0);
            self.uvw = Vector3d(0, 0, 0);
            self.ang = Vector3d(0, 0, 0);
            self.pqr = Vector3d(0, 0, 0);
        }

        //Save this timestep's values for logging purposes
        self.update_history(self);

        //Calculate and store the vortex produced by this bird on this time step
        self.vortex_buffer_left = Vortex(self, -1);
        self.vortex_buffer_right = Vortex(self, 1);
    }

    //Save values for logging and plotting
    void update_history(const Bird & b){
        Bird & self = *this;
        self.prev_xyz.push_back(b.xyz);
        self.prev_uvw.push_back(b.uvw);
        self.prev_ang.push_back(b.ang);
        self.prev_pqr.push_back(b.pqr);

        self.ALPHA_L.push_back(b.alpha_l);
        self.ALPHA_R.push_back(b.alpha_r);
        self.BETA_L.push_back(b.beta_l);
        self.BETA_R.push_back(b.beta_r);
    }

    /*
      updates the vortex positions according to the
      local induction approximation (LIA), which describes the motion
      of vortex lines
    */
    void update_vortex_positions(Vorticies & vortices, double h){
        Bird & self = *this;
        for (size_t i : range(1, len(vortices)-2)){

            //tangent vectors
            Vector3d t_minus = vortices[i].pos - vortices[i-1].pos;
            Vector3d t = vortices[i+1].pos - vortices[i].pos;

            //length of tangent vectors
            double l_t_minus = t_minus.norm();
            double l_t = t.norm();

            double theta;

            //checking the limits of arccos
            if (dot(t_minus, t)/(l_t_minus * l_t) <= -1.0){
                theta = arccos(-1.0);
            }
            if (dot(t_minus, t)/(l_t_minus * l_t) >= 1.0){
                theta = arccos(1.0);
            }

            else{
                theta = arccos(dot(t_minus, t)/(l_t_minus * l_t));
            }

            //normal vector
            Vector3d n = (t - t_minus);
            n = n/n.norm();
            //binormal vector
            Vector3d b = cross(t, n);

            //normalize n and b, (protecting against divide by zero)
            if (n.norm() > 0.0){
                n = n/n.norm();
            }
            if (b.norm() > 0.0){
                b = b/b.norm();
            }

            //Update vortex strength based on decay equation
            Vortex vor = vortices[i];
            vor.t ++;
            if(vor.decaying){
              vor.gamma = vor.gamma_0 * pow((vor.t_decay/vor.t), vor.decay_exponent);
            }
            //Replace with probability if we decide not to make it deterministic
            else{
              if(vor.t > vor.t_decay){
                vor.decaying = true;
              }
            }
            double gamma = vor.gamma;
            //distance from last point
            double epsilon = l_t_minus;

            //vortex's new position after LIA calculation
            Vector3d pos = self.take_vortex_time_step(vortices[i].pos, gamma, epsilon, b, theta, h);

            //update distance travelled and position for this time step
            vortices[i].dist_travelled += (vortices[i].pos - pos).norm();
            vortices[i].pos = pos;
        }
    }

    /*
      If we are on a time step where a point on the vortex line should be added,
      then add that point.
    */
    void shed_vortices(){
        Bird & self = *this;
        self.VORTICES_RIGHT.push_back(self.vortex_buffer_right);
        self.VORTICES_LEFT.push_back(self.vortex_buffer_left);
    }

    //return forces and torques on this bird due to vortices it interacts with
    VortexForces vortex_forces(Vorticies & vortices){
        Bird & self = *this;
        //forces in each direction
        Vector3d fuvw;
        //torques in each direction
        Vector3d tuvw;
        //drag and areas
        double D, A;

        for (Vortex & vortex : vortices){
            //get velocity of moving vortex air in the bird's frame for the
            //left and right wing.
            auto bird_pair = vortex.bird_vel(self);
            Vector3d L = bird_pair.first;
            Vector3d R = bird_pair.second;

            //calculate up and down forces for left wing
            //Approximate area of left wing
            A = self.Xl * self.Yl;
            //Calculate drag in the w direction
            D = sign(L[2]) * self.Cd * A * (self.rho * sqr(L[2]))/2.0;
            fuvw[2] += D;
            tuvw[0] -= D * self.Xl/2.0;

            //calculate up and down forces for right wing
            //Approximate area of right wing
            A = self.Xl * self.Yl;
            //Calculate drag on right wing
            D = sign(R[2]) * self.Cd * A * (self.rho * sqr(R[2]))/2.0;
            fuvw[2] += D;
            tuvw[0] += D * self.Xl/2.0;
        }
        //return forces and torques in each direction
        return VortexForces{fuvw, tuvw};
    }

    /*
      Find the n birds closest to this bird.
      Birds only consider the closest ~7 birds when making decisions
    */
    std::vector<Bird *> n_nearest(Birds & birds, size_t max_observable_birds){
        Bird & self = *this;
        std::vector<int> arange(birds.size());
        for(size_t i = 0; i < birds.size(); i++){
            arange.at(i) = i;
        }

        //Calculate distances from this bird to all other birds
        std::vector<double> dists(arange.size());
        for(size_t i = 0; i < arange.size(); i++){
            Bird & other = birds[arange[i]];
            dists[i] = (self.xyz - other.xyz).norm();
            /* dists[i] = sqrt(sqr(self.x - other.x) + sqr(self.y - other.y) + sqr(self.z - other.z)); */
        }

        //Sort these in order of nearest to farthest from this bird
        std::stable_sort(arange.begin(),arange.end(), [&](int b1, int b2){
            return dists[b1] < dists[b2];
        });
        assert(dists[arange[0]] == 0 && "the smallest distance should be zero");
        std::vector<Bird *> result;

        //Collect and return the n nearest birds from the sorted list
        for(size_t i = 1; i < std::min(max_observable_birds + 1, arange.size()); i++){
            result.push_back(&birds[arange[i]]);
        }
        return result;
    }

    /*
      This is an RK4 differential equations solver.
      It numerically estimates the differential equations that represent the
      time evolution of position, orientation, velocity, and angular velocity
    */
    Vector3d take_time_step(DiffEqType ddt, Vector3d y, double h){
        Bird & self = *this;
        Vector3d k1 = h * ddt(y, self);
        Vector3d k2 = h * ddt(y + 0.5 * k1, self);
        Vector3d k3 = h * ddt(y + 0.5 * k2, self);
        Vector3d k4 = h * ddt(y + k3, self);

        return y + (1.0 / 6.0)*(k1 + (2.0 * k2) + (2.0 * k3) + k4);
    }

    /*
      This is the RK4 solver for the LIA for motion of the vortices
    */
    Vector3d take_vortex_time_step(Vector3d pos,double gamma, double epsilon, Vector3d b,double theta,double h){
        Vector3d k1 = h * drdt(gamma, epsilon, b, theta);
        Vector3d k2 = h * drdt(gamma, epsilon, b, theta);
        Vector3d k3 = h * drdt(gamma, epsilon, b, theta);
        Vector3d k4 = h * drdt(gamma, epsilon, b, theta);

        return pos + (1.0 / 6.0)*(k1 + (2.0 * k2) + (2.0 * k3) + k4);
    }

    bool operator < (const Bird & other)const{
        const Bird & self = *this;
        return self.xyz[0] < other.xyz[0];
    }
};
using Birds = std::vector<Bird>;
