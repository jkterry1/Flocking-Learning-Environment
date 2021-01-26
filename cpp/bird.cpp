#include <array>
#include <vector>
#include <queue>
#include <cassert>
#include "vortex.hpp"
#include "utils.hpp"
#include "DiffEqs.hpp"

using PrevValues = std::vector<double>;
using VortexForces = std::array<double,6>;
using Vorticies = std::vector<Vortex>;
typedef Vector3d (DiffEqType)(Vector3d uvw, Bird & bird);

struct BirdInit{
    double u,v,w;
    double p,q,r;
    double x,y,z;
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
        Cl, lift coefficient
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
    static constexpr double Zl = 0.15;

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
    Vector3d angles()const{return Vector3d(theta,phi,psi);}

    /*
    position
        x, y, z
    */
    double x,y,z;
    Vector3d xyz()const{return Vector3d(x,y,z);}

    /*
    wing orientation
    */
    double alpha_l;// = alpha_l
    double beta_l;// = beta_l
    double alpha_r;// = alpha_r
    double beta_r;// = beta_r

    /*
    Observation variables:
        F, net force
        T, net torque
    */
    Vector3d F;
    Vector3d T;

    Vortex vortex_buffer_left;
    Vortex vortex_buffer_right;
    Vorticies VORTICES_RIGHT;
    Vorticies VORTICES_LEFT;// = [Vortex(self, -1)]
    double vortex_force_u = 0, vortex_force_v = 0, vortex_force_w = 0;//[0.0, 0.0, 0.0]
    double vortex_torque_u = 0, vortex_torque_v = 0, vortex_torque_w = 0;//[0.0, 0.0, 0.0]

    double thrust;
    //disable copy and assignment

    /*
    keep track of old values
    */
    PrevValues U;// = [u]
    PrevValues V;// = [v]
    PrevValues W;// = [w]

    PrevValues X;// = [x]
    PrevValues Y;// = [y]
    PrevValues Z;// = [z]

    PrevValues P;// = [p]
    PrevValues Q;// = [q]
    PrevValues R;// = [r]

    PrevValues THETA;// = [theta]
    PrevValues PHI;// = [phi]
    PrevValues PSI;// = [psi]

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
        // double v = 0.0, w = 0.0,
        // p = 0.0, q = 0.0, r = 0.0,
        // theta = 0.0, phi = 0.0, psi = 0.0,
        double alpha_l = 0.0, beta_l = 0.0, alpha_r = 0.0, beta_r = 0.0;
        // x = 0.0;//, y = 0.0, z = 0.0;


        // self.g = -9.8
        // self.rho = 1.225

        // self.m = 5.0        //goose is ~5kg
        // self.Cl_max = 1.6   //experimental value
        // self.Cd = 0.3       //experimental value
        // self.S = 0.62
        // self.Xl = 0.80     //approximate dimensions
        // self.Yl = 0.35
        // self.Zl = 0.15

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

    	self.F = Vector3d(0.0, 0.0, 0.0);
    	self.T = Vector3d(0.0, 0.0, 0.0);
        /*
        wing orientation
        */
        self.alpha_l = alpha_l;
        self.beta_l = beta_l;
        self.alpha_r = alpha_r;
        self.beta_r = beta_r;

        /*
        Observation variables:
            F, net force
            T, net torque
        */
        /*
        keep track of old values
        */
        self.VORTICES_RIGHT = Vorticies{Vortex(self, 1)};
        self.VORTICES_LEFT = Vorticies{Vortex(self, -1)};

        self.update_history(self);
    }

    void update(double thrust, double h, Vorticies & vortices){
        Bird & self = *this;
        self.thrust = thrust;

        VortexForces a = self.vortex_forces(vortices);
        self.vortex_force_u = a[0];
        self.vortex_force_v = a[1];
        self.vortex_force_w = a[2];
        self.vortex_torque_u = a[3];
        self.vortex_torque_v = a[4];
        self.vortex_torque_w = a[5];
        //print()
        //print("Updating uvw")
        Vector3d uvw = self.take_time_step(duvwdt, self.uvw(), h);
        up(self.u,self.v,self.w) = uvw;

        Vector3d pqr = self.take_time_step(dpqrdt, self.pqr(), h);
        up(self.p,self.q,self.r) = pqr;

        Vector3d angles = self.take_time_step(danglesdt, self.angles(), h);
        up(self.theta,self.phi,self.psi) = angles;

        Vector3d xyz = self.take_time_step(dxyzdt, self.xyz(), h);
        up(self.x,self.y,self.z) = xyz;

        if( self.z <= 0){
            self.z = 0;
            u = 0;
            v = 0;
            w = 0;
            p = 0;
            q = 0;
            r = 0;
        }

        self.update_history(self);
        // Shed a vortex
        // if u > 0 || v > 0 || w > 0{
        //     //right, left
        //     self.vortex_buffer = (Vortex(self, 1), Vortex(self, -1))
        self.vortex_buffer_left = Vortex(self, -1);
        self.vortex_buffer_right = Vortex(self, 1);
    }
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

    void update_vortex_positions(Vorticies & vortices, double h){
        Bird & self = *this;
        for (size_t i : range(1, len(vortices)-2)){
            //tangent vectors
            Vector3d t_minus = vortices[i].pos - vortices[i-1].pos;
            Vector3d t = vortices[i+1].pos - vortices[i].pos;
            Vector3d t_plus = vortices[i+2].pos - vortices[i+1].pos;

            double l_t_minus = t_minus.norm();
            double l_t = t.norm();
            double theta;
            //print(dot(t_minus, t)/(l_t_minus * l_t))

            // if np.linalg.norm(t_minus) > 0.0:
            //     t_minus = t_minus/np.linalg.norm(t_minus)
            // if np.linalg.norm(t) > 0.0:
            //     t = t/np.linalg.norm(t)
            // if np.linalg.norm(t_plus) > 0.0:
            //     t_plus = t_plus/np.linalg.norm(t_plus)

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
            Vector3d n = (cross(t, t_plus)/(1 * dot(t, t_plus))) - (cross(t_minus, t)/(1 + dot(t_minus, t)));
            //binormal vector
            Vector3d b = cross(t, n);

            if (n.norm() > 0.0){
                n = n/n.norm();
            }

            if (b.norm() > 0.0){
                b = b/b.norm();
            }
            //strength
            double gamma = vortices[i].gamma;
            //distance from last point
            double epsilon = l_t_minus;

            Vector3d pos = self.take_vortex_time_step(vortices[i].pos, gamma, epsilon, b, theta, h);

            //print(abs(vortices[i].pos - pos))
            vortices[i].dist_travelled += (vortices[i].pos - pos).norm();
            vortices[i].pos = pos;
        }
    }

    void shed_vortices(){
        Bird & self = *this;
        // x = self.vortex_buffer[0].x;
        // y = self.vortex_buffer[0].y;
        // z = self.vortex_buffer[0].z;
        //print("Shedding vortex at ", [x,y,z])
        self.VORTICES_RIGHT.push_back(self.vortex_buffer_right);
        self.VORTICES_LEFT.push_back(self.vortex_buffer_left);
    }
    //returns fu, fv, fw, tu, tv, tw
    VortexForces vortex_forces(Vorticies & vortices){
        Bird & self = *this;
        double fu=0, fv=0, fw=0;
        double tu=0, tv=0, tw=0;// = [0.0, 0.0, 0.0]
        double D, Aw, A;
        double u,v,w;
        for (Vortex & vortex : vortices){
            //returns velocities on left and right wing
            auto bird_pair = vortex.bird_vel(self);
            Vector3d L = bird_pair.first;
            Vector3d R = bird_pair.second;

            // print()
            // print("bird ", self)
            // print("vort ", vortex)
            // print("L ", L)
            // print("R ", R)

            //calculate up and down forces
            //left wing
            up(u, v, w) = L;
            Aw = self.Xl * self.Yl;
            D = sign(w) * self.Cd * Aw * (self.rho * sqr(w))/2.0;
            fw += D;
            tu -= D * self.Xl/2.0;

            //right wing
            up(u, v, w) = R;
            A = self.Xl * self.Yl;
            D = sign(w) * self.Cd * A * (self.rho * sqr(w))/2.0;
            fw += D;
            tu += D * self.Xl/2.0;
        }
        return VortexForces{fu, fv, fw, tu, tv, tw};
    }

    std::vector<Bird *> n_nearest(Birds & birds, size_t max_observable_birds){
        Bird & self = *this;
        std::vector<int> arange(birds.size());
        for(size_t i = 0; i < birds.size(); i++){
            arange.at(i) = i;
        }
        std::vector<double> dists(arange.size());
        for(size_t i = 0; i < arange.size(); i++){
            Bird & other = birds[arange[i]];
            dists[i] = sqrt(sqr(self.x - other.x) + sqr(self.y - other.y) + sqr(self.z - other.z));
        }
        std::stable_sort(arange.begin(),arange.end(), [&](int b1, int b2){
            return dists[b1] < dists[b2];
        });
        assert(dists[arange[0]] == 0 && "the smallest distance should be zero");
        std::vector<Bird *> result;
        for(size_t i = 1; i < std::min(max_observable_birds + 1, arange.size()); i++){
            result.push_back(&birds[arange[i]]);
        }
        return result;
    }

    Vector3d take_time_step(DiffEqType ddt, Vector3d y, double h){
        Bird & self = *this;
        Vector3d k1 = h * ddt(y, self);
        Vector3d k2 = h * ddt(y + 0.5 * k1, self);
        Vector3d k3 = h * ddt(y + 0.5 * k2, self);
        Vector3d k4 = h * ddt(y + k3, self);

        return y + (1.0 / 6.0)*(k1 + (2.0 * k2) + (2.0 * k3) + k4);
    }
    Vector3d take_vortex_time_step(Vector3d pos,double gamma, double epsilon, Vector3d b,double theta,double h){
        // Bird & self = *this;
        // TODO: this formula is weird. You calculate the same value 4 times, it never changes. This is almost certainly wrong
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
