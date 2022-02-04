#include "types.hpp"
#include "utils.hpp"
#include "vortex.hpp"
#include "DiffEqs.hpp"
#include "bird.cpp"
#include "DiffEqs.cpp"
#include "vortex.cpp"
#include <iostream>
#include <stdlib.h>

using namespace std;

using Birds = std::vector<Bird>;
struct Flock{

    double h;
    double t;
    int N;
    double max_r;
    bool LIA;
    double energy_reward;
    double forward_reward;
    double crash_reward;
    Birds birds;
    BirdInits starting_conditions;
    bool derivatives;
    double max_dist;
    double thrust_limit;
    double wing_action_limit_alpha;
    double wing_action_limit_beta;
    double height_limit;
    unsigned int random_seed;
    bool include_vortices;
    std::vector<double> limits;
    Flock(  int N,
            double h,
            double t,
            double energy_reward,
            double forward_reward,
            double crash_reward,
            BirdInits init_vals,
            bool LIA,
            bool derivatives,
            double thrust_limit,
            double wing_action_limit_alpha,
            double wing_action_limit_beta,
            unsigned int random_seed,
            bool include_vortices
         ){
        Flock & self = *this;

        //Parameter descriptions available in flocking_env
        self.h = h;
        self.t = t;
        self.N = N;
        self.LIA = LIA;
        self.include_vortices = include_vortices;

        self.max_r = 1.0;

        self.energy_reward = energy_reward;
        self.forward_reward = forward_reward;
        self.crash_reward = crash_reward;

        self.thrust_limit = thrust_limit;
        self.wing_action_limit_alpha = wing_action_limit_alpha;
        self.wing_action_limit_beta = wing_action_limit_beta;

        self.random_seed = random_seed;

        for (size_t i : range(N)){
            self.birds.emplace_back(init_vals[i]);
        }
        self.starting_conditions = init_vals;

        self.derivatives = derivatives;

        self.max_dist = 0.0;

        self.height_limit = 200.0;

        //force limit
        self.limits.push_back(20.0*9.8*birds[0].m); //0
        self.limits.push_back(20.0*9.8*birds[0].m); //1
        self.limits.push_back(20.0*9.8*birds[0].m); //2

        //Torque limit
        self.limits.push_back(100.0*birds[0].Xl); //3
        self.limits.push_back(100.0*birds[0].Xl); //4
        self.limits.push_back(100.0*birds[0].Xl); //5

        //Height limit
        self.limits.push_back(self.height_limit); //6

        //Orientation Limits, 2pi rad
        self.limits.push_back(6.3); //7
        self.limits.push_back(6.3); //8
        self.limits.push_back(6.3); //9

        double limit_alpha_low = -25.0 * PI/180.0;
        double limit_alpha_high = 15.0 * PI/180.0;
        double limit_beta_low = -30.0 * PI/180.0;
        double limit_beta_high = 40.0 * PI/180.0;

        //wing alpha low/high
        self.limits.push_back(limit_alpha_low); //10
        self.limits.push_back(limit_alpha_high); //11

        //wing beta low/high
        self.limits.push_back(limit_beta_low); //12
        self.limits.push_back(limit_beta_high); //13

        //velocity limit, 50 m/s
        self.limits.push_back(50.0); //14
        self.limits.push_back(50.0); //15
        self.limits.push_back(50.0); //16

        //angular velocity limits,rad/s
        self.limits.push_back(20.0); //17
        self.limits.push_back(20.0); //18
        self.limits.push_back(20.0); //19

        //other bird relative position
        self.limits.push_back(500.0); //20
        self.limits.push_back(500.0); //21
        self.limits.push_back(500.0); //22

        //other bird's relative orientation
        self.limits.push_back(2.0 * 6.3); //23
        self.limits.push_back(2.0 * 6.3); //24
        self.limits.push_back(2.0 * 6.3); //25

        //other bird's relative velocity
        self.limits.push_back(100.0); //26
        self.limits.push_back(100.0); //27
        self.limits.push_back(100.0); //28

        // seed random number generator for noise()
        srand(random_seed);
    }

    //Restores birds to their initial conditions
    void reset(){
        Flock & self = *this;
        for (size_t i : range(self.N)){
            self.birds[i] = Bird(self.starting_conditions[i]);
        }
    }

    //Update the bird's properties for the next timestep given the parameter action.
    void update_bird(EnvAction action, int agent){
        Flock & self = *this;
        double thrust = action[0];
        Bird & bird = self.birds[agent];

        //Update the wing positions, then make the calculations for the next timestep.
        self.update_angles(action, agent);
        Vorticies vortices;
        if (self.include_vortices){
          vortices = self.get_vortices(bird);
        }
        bird.update(thrust, self.h, vortices);
    }

    //Returns the done status and reward for this bird for this timestep.
    std::pair<bool, double> get_done_reward(EnvAction & action, int agent){
        Flock & self = *this;
        double reward = 0;
        bool done = false;
        Bird & bird = self.birds[agent];

        int i_last = len(bird.X)-2;

        /*
        The bird is rewarded for the distance it has travelled.
        */
        reward += self.forward_reward * (bird.x - bird.X[i_last]);

        /*
        Energy used by the bird is proportional to the birds thrust and the net force on the bird
         times the distance it traveled (work = force * distance)
        */
        reward += self.energy_reward * action[0] * self.h * abs(bird.u);

        //If the bird has crashed, we consider it done and punish it for crashing.
        if (self.crashed(bird)){
            done = true;
            reward += self.crash_reward;
        }

        //This can be changed depending on your goals.
        //Gives the bird a destination, flying 500m forward ends the simulation.
        // if (bird.x > 500.0){
        //     done = true;
        // }

        return std::make_pair(done, reward);
    }

    /*
    Checks to see if the bird has crashed in some way.
    */
    bool crashed(const Bird & bird){
        Flock & self = *this;

        //Checks if bird ahs hit the ground, or gone too high
        if (bird.z <= 0 || bird.z > self.height_limit){
            return true;
        }

        //velocity limit of 40 m/s
        double v_lim = 40.0;
        if (abs(bird.u) > v_lim || abs(bird.v) > v_lim ||  abs(bird.w) > v_lim){
            return true;
        }

        //This is the rotation speed limit.
        //If the bird starts rotating too quickly,
        //it is considered to have crashed.
        double lim = 2*PI;
        if (abs(bird.p) > lim || abs(bird.q) > lim ||  abs(bird.r) > lim){
            return true;
        }

        /*
        Checks if the current bird is within a certain radius of another bird.
        This is considered crashing into the other bird.
        */
        bool crash = false;
        for (int b : range(len(self.birds))){
            const Bird & other = self.birds[b];
            if (&other != &bird){
                double dist = (bird.xyz() - other.xyz()).norm();
                if (dist < bird.Xl/2.0){
                    crash = true;
                }
            }
        }
        return crash;
    }

    /*
    Returns all active vortices generates by the parameter bird.
    */
    Vorticies get_vortices(Bird & curr){
        Flock & self = *this;
        Vorticies vortices;
        for (size_t b : range(len(self.birds))) {
            const Bird & bird = self.birds[b];

            if (&bird != &curr){
                for (const Vorticies & vorts : {bird.VORTICES_LEFT, bird.VORTICES_RIGHT}){
                    size_t i = 0;
                    /*
                    forward motion for birds is in the x direction.

                    Looking at all vortices another bird has produced,
                    the current bird interacts with only vortices that are
                    either in its exact x location or ahead of it, and
                    chooses the closest of those to consider.

                    If the closest vortex ahead of the current bird is
                    not within max_r distance from the center of the vortex,
                    it is not considered.
                    */
                    while (i < len(vorts) && vorts[i].x() < curr.x){
                        i = i+1;
                    }
                    const Vortex & v = vorts[i];

                    //Move through vortices until the first vortex that is in
                    //front of the current bird.
                    if (i < len(vorts) && v.x() >= curr.x){
                        //Determine if the bird is too far from the
                        //center of this vortex to be affected.
                        double r = sqrt(sqr(curr.y - v.y()) + sqr(curr.z - v.z()));
                        if (r < self.max_r){
                            vortices.push_back(v);
                        }
                    }
                }
            }
        }
        return vortices;
    }

    double noise(double max){
      return (max * (rand()) / (RAND_MAX));
    }

    /*
    Returns the observation for one bird.
    Refer to flocking_env or the readme for a descriptiion of the contents
    of the observation.
    */
    Observation get_observation(int agent, int max_observable_birds){
        Flock & self = *this;
        Vector3d force = self.birds[agent].F;
        Vector3d torque = self.birds[agent].T;

        Bird & bird = self.birds[agent];
        Observation obs;
        extend(obs, ((force)/self.limits[0]) + Vector3d{1.0,1.0,1.0}/2.0); //0,1,2
        extend(obs, (torque + self.limits[3])/(2.0 * self.limits[3]));//3,4,5
        extend(obs, {((bird.z)/self.limits[6] + 1.0)/2.0});//6
        extend(obs, {((bird.phi)/self.limits[7] + 1.0)/2.0,//7
                      ((bird.theta)/self.limits[8] + 1.0)/2.0,//8
                      ((bird.psi)/self.limits[9] + 1.0)/2.0});//9
        extend(obs, {(((bird.alpha_l) - self.limits[10])/(self.limits[11] - self.limits[10])),//10
                      (((bird.beta_l) - self.limits[12])/(self.limits[13] - self.limits[12])),//11
                      (((bird.alpha_r) - self.limits[10])/(self.limits[11] - self.limits[10])),//12
                      (((bird.beta_r) - self.limits[12])/(self.limits[13] - self.limits[12]))});//13
        if(derivatives){
          extend(obs, {((bird.u+noise(.01))/self.limits[14] + 1.0)/2.0,//14
                        ((bird.v+noise(.01))/self.limits[15] + 1.0)/2.0,//15
                        ((bird.w+noise(.01))/self.limits[16] + 1.0)/2.0});//16
          extend(obs, {((bird.p+noise(.01))/self.limits[17] + 1.0)/2.0,//17
                        ((bird.q+noise(.01))/self.limits[18] + 1.0)/2.0,//18
                        ((bird.r+noise(.01))/self.limits[19] + 1.0)/2.0});//19
        }
        std::vector<Bird *> nearest = bird.n_nearest(self.birds, max_observable_birds);
        for (Bird * otherp : nearest){
	        Bird & other = *otherp;
            extend(obs, {(((other.x - bird.x)+noise(.01))/self.limits[20] + 1.0)/2.0,//20
                          (((other.y - bird.y)+noise(.01))/self.limits[21] + 1.0)/2.0,//21
                          (((other.z - bird.z)+noise(.01))/self.limits[22] + 1.0)/2.0});//22
            extend(obs, {(((other.phi - bird.psi)+noise(.01))/self.limits[23] + 1.0)/2.0,//23
                          (((other.theta - bird.theta)+noise(.01))/self.limits[24] + 1.0)/2.0,//24
                          (((other.psi - bird.psi)+noise(.01))/self.limits[25] + 1.0)/2.0});//25
            if(derivatives){
              extend(obs, {(((other.u - bird.u)+noise(.01))/self.limits[26] + 1.0)/2.0,//26
                            (((other.v - bird.v)+noise(.01))/self.limits[27] + 1.0)/2.0,//27
                            (((other.w - bird.w)+noise(.01))/self.limits[28] + 1.0)/2.0 });//28
            }
        }
        return obs;
    }

    /*
    Adds a new point on the vortex line based on the bird's current state.
    Checks for expired vortices,
    Vortices "expire" after one second and are removed from the simulation.
    If LIA is true, all active vortices are moved according to their LIA evolution
    */
    void update_vortices(double vortex_update_frequency){
        Flock & self = *this;
        for (size_t b : range(len(self.birds))){
            Bird & bird = self.birds[b];
            //Adds new vortex points
            bird.shed_vortices();

            if (self.LIA){
                //Updates all vortex positions based on LIA
                bird.update_vortex_positions(bird.VORTICES_RIGHT, self.h*vortex_update_frequency);
                bird.update_vortex_positions(bird.VORTICES_LEFT, self.h*vortex_update_frequency);
            }

            //remove expired vortices, vortices are only active for one second
            if (len(bird.VORTICES_RIGHT) > 1.0/(self.h*vortex_update_frequency)){
                pop0(bird.VORTICES_LEFT);
                pop0(bird.VORTICES_RIGHT);
            }
        }
    }

    /*
    Given the action for this timestep, updates the bird's wing angles.
    */
    void update_angles(EnvAction action,int agent){
        Flock & self = *this;
        Bird & bird = self.birds[agent];

        //The limits for wing rotation in radians
        //Starling:
        double limit_alpha_low = -25.0 * PI/180.0;
        double limit_alpha_high = 15.0 * PI/180.0;
        double limit_beta_low = -30.0 * PI/180.0;
        double limit_beta_high = 40.0 * PI/180.0;

        //Geese
        /*
        double limit_alpha_low = -90.0 * PI/180.0;
        double limit_alpha_high = 90.0 * PI/180.0;
        double limit_beta_low = -90.0 * PI/180.0;
        double limit_beta_high = 90.0 * PI/180.0;
        */

        /*
        Calculate the new alpha angle for each wing, taking the limits into account.
        The wing angles cannot exceed the maximums or minimums,
        and if an action tries to exceed these values, the wings will just
        remain at the max or min value.
        */
        double new_al = bird.alpha_l + action[1];
        if (new_al > limit_alpha_high)
            new_al = limit_alpha_high;
        if (new_al < limit_alpha_low)
            new_al = limit_alpha_low;
        bird.alpha_l = new_al;

        double new_bl = bird.beta_l + action[2];
        if (new_bl > limit_beta_high)
            new_bl = limit_beta_high;
        if (new_bl < limit_beta_low)
            new_bl = limit_beta_low;
        bird.beta_l = new_bl;

        double new_ar = bird.alpha_r + action[3];
        if (new_ar > limit_alpha_high)
            new_ar = limit_alpha_high;
        if (new_ar < limit_alpha_low)
            new_ar = limit_alpha_low;
        bird.alpha_r = new_ar;

        double new_br = bird.beta_r + action[4];
        if (new_br > limit_beta_high)
            new_br = limit_beta_high;
        if (new_br < limit_beta_low)
            new_br = limit_beta_low;
        bird.beta_r = new_br;

    }


    Birds get_birds(){
        return this->birds;
    }
};
