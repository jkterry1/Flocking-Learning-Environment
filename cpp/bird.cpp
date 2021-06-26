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
using VortexForces = std::array<double,6>;
using Vorticies = std::vector<Vortex>;
typedef Vector3d (DiffEqType)(Vector3d uvw, Bird & bird);

using namespace std;

struct BirdInit{
    //velocities in the u (forward), v (out from the right wing), and w (up) directions
    double u,v,w;

    //angular velocities around the u, v, and w axes
    double p,q,r;

    //position along x, y, and z axes
    double x,y,z;

    //angle bird is rotated around the inertial x, y, and z axes
    double theta;
    double phi;
    double psi;

    using dbl = double;
    BirdInit(dbl x, dbl y, dbl z, dbl u, dbl v, dbl w, dbl p, dbl q, dbl r, dbl theta, dbl phi, dbl psi){
        BirdInit & self = *this;
        self.x = x;
        self.y = y;
        self.z = z;
        self.u = u;
        self.v = v;
        self.w = w;
        self.p = p;
        self.q = q;
        self.r = r;
        self.theta = theta;
        self.phi = phi;
        self.psi = psi;
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

    // fixed body frame variables:
    /*
    velocity:
        u points out the front of the bird
        v points out of the right wing
        w points up through the center of the bird
    */
    double u,v,w;
    Vector3d uvw()const{return Vector3d(u,v,w);}

    /*
    angular velocity:
        p is the rotation about the axis coming out the front of the bird
        q is the rotation about teh axis coming out the right wing of the bird
        r is the rotation about the axis coming out the top of the bird
    */
    double p,q,r;
    Vector3d pqr()const{return Vector3d(p,q,r);}

    // intertial frame variables
    /*
    euler angles:
        theta is the rotated angle about the intertial x axis
        phi is the rotated angle about the intertial y axis
        psi is the rotated angle about the intertial z axis
    */
    double theta;
    double phi;
    double psi;
    Vector3d angles()const{return Vector3d(phi, theta, psi);}

    /*
    position
        x, y, z
    */
    double x,y,z;
    Vector3d xyz()const{return Vector3d(x,y,z);}

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
    double vortex_force_u = 0, vortex_force_v = 0, vortex_force_w = 0;
    double vortex_torque_u = 0, vortex_torque_v = 0, vortex_torque_w = 0;

    //The bird's thrust, which comes entirely from the bird's chosen action
    double thrust;

    /*
    keep track of old values
    */
    PrevValues U;
    PrevValues V;
    PrevValues W;

    PrevValues X;
    PrevValues Y;
    PrevValues Z;

    PrevValues P;
    PrevValues Q;
    PrevValues R;

    PrevValues THETA;
    PrevValues PHI;
    PrevValues PSI;

    PrevValues ALPHA_L;
    PrevValues ALPHA_R;
    PrevValues BETA_L;
    PrevValues BETA_R;

    Bird() = default;
    Bird(const BirdInit & init){
        Bird & self = *this;
        self.x = init.x;
        self.y = init.y;
        self.z = init.z;
        self.u = init.u;
        self.v = init.v;
        self.w = init.w;
        self.p = init.p;
        self.q = init.q;
        self.r = init.r;
        self.theta = init.theta;
        self.phi = init.phi;
        self.psi = init.psi;

        double alpha_l = 0.0, beta_l = 0.0, alpha_r = 0.0, beta_r = 0.0;

        /*
        Moments of Inertia
        */
        self.Ixx = self.m * sqr(self.Xl);
        self.Iyy = self.m * sqr(self.Yl);
        self.Izz = self.m * sqr(self.Zl);
        self.Ixz = 0.5 * self.m * sqr(self.Zl);    //approximation
        self.inertia = matrix(self.Ixx, 0.0, self.Ixz,
                                    0.0, self.Iyy, 0.0,
                                    self.Ixz, 0.0, self.Izz);


        //Total force and torque on this bird
        self.F = Vector3d(0.0, 0.0, 0.0);
      	self.T = Vector3d(0.0, 0.0, 0.0);

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
        self.vortex_force_u = vf[0];
        self.vortex_force_v = vf[1];
        self.vortex_force_w = vf[2];
        self.vortex_torque_u = vf[3];
        self.vortex_torque_v = vf[4];
        self.vortex_torque_w = vf[5];

        //Calculate and update velocities for the next time step using the diffeq solver
        Vector3d uvw = self.take_time_step(duvwdt, self.uvw(), h);
        up(self.u,self.v,self.w) = uvw;
        //cout<<self.w << " ";

        //Calculate and update angular velocities for the next time step
        Vector3d pqr = self.take_time_step(dpqrdt, self.pqr(), h);
        up(self.p,self.q,self.r) = pqr;

        //calculate and update orientation for the next time step
        Vector3d angles = self.take_time_step(danglesdt, self.angles(), h);
        angles[0] = fmod(angles[0], (2.0*self.pi));
        angles[1] = fmod(angles[1], (2.0*self.pi));
        angles[2] = fmod(angles[2], (2.0*self.pi));
        up(self.phi, self.theta, self.psi) = angles;
        //cout<<"angles "<<angles<<"\n";

        //calculate and update position for the next time step
        Vector3d xyz = self.take_time_step(dxyzdt, self.xyz(), h);
        up(self.x,self.y,self.z) = xyz;

        /*
          Check if the bird has crashed into the ground.
          If so, set vertical position, all velocities,
          and angular velocities to 0 (bird no longer moving)
        */
        if(self.z <= 0){
            self.z = 0;
            u = 0;
            v = 0;
            w = 0;
            p = 0;
            q = 0;
            r = 0;
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
        self.U.push_back(b.u);
        self.V.push_back(b.v);
        self.W.push_back(b.w);

        self.X.push_back(b.x);
        self.Y.push_back(b.y);
        self.Z.push_back(b.z);

        self.P.push_back(b.p);
        self.Q.push_back(b.q);
        self.R.push_back(b.r);

        self.THETA.push_back(b.theta);
        self.PHI.push_back(b.phi);
        self.PSI.push_back(b.psi);

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
            //Vector3d t_plus = vortices[i+2].pos - vortices[i+1].pos;

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
            //Vector3d n = (cross(t, t_plus)/(1 * dot(t, t_plus))) - (cross(t_minus, t)/(1 + dot(t_minus, t)));
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
        double fu=0, fv=0, fw=0;
        //torques in each direction
        double tu=0, tv=0, tw=0;
        //drag and areas
        double D, A;
        //velocities in each direction
        double u,v,w;

        for (Vortex & vortex : vortices){
            //get velocity of moving vortex air in the bird's frame for the
            //left and right wing.
            auto bird_pair = vortex.bird_vel(self);
            Vector3d L = bird_pair.first;
            Vector3d R = bird_pair.second;

            //calculate up and down forces
            //left wing
            up(u, v, w) = L;
            //Approximate area of left wing
            A = self.Xl * self.Yl;
            //Calculate drag in the w direction
            D = sign(w) * self.Cd * A * (self.rho * sqr(w))/2.0;
            fw += D;
            tu -= D * self.Xl/2.0;

            //right wing
            up(u, v, w) = R;
            //Approximate area of right wing
            A = self.Xl * self.Yl;
            //Calculate drag on right wing
            D = sign(w) * self.Cd * A * (self.rho * sqr(w))/2.0;
            fw += D;
            tu += D * self.Xl/2.0;
        }
        //return forces and torques in each direction
        //cout << fu << " " << fv << " " << fw << "\n";
        //cout << tu << " " << tv << " " << tw << "\n";
        return VortexForces{fu, fv, fw, tu, tv, tw};
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
            dists[i] = sqrt(sqr(self.x - other.x) + sqr(self.y - other.y) + sqr(self.z - other.z));
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
        return self.x < other.x;
    }
};
using Birds = std::vector<Bird>;
