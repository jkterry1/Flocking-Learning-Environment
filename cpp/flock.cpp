#include "types.hpp"
#include "utils.hpp"
#include "vortex.hpp"
#include "DiffEqs.hpp"
#include "bird.cpp"
#include "DiffEqs.cpp"
#include "vortex.cpp"

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
    Flock(  int N,
            double h,
            double t,
            double energy_reward,
            double forward_reward,
            double crash_reward,
            BirdInits init_vals,
            bool LIA
         ){
        Flock & self = *this;

        //Parameter descriptions available in flocking_env
        self.h = h;
        self.t = t;
        self.N = N;
        self.LIA = LIA;

        self.max_r = 1.0;

        self.energy_reward = energy_reward;
        self.forward_reward = forward_reward;
        self.crash_reward = crash_reward;


        for (size_t i : range(N)){
            self.birds.emplace_back(init_vals[i]);
        }
        self.starting_conditions = init_vals;
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
        Vorticies vortices = self.get_vortices(bird);
        bird.update(thrust, self.h, vortices);
    }

    //Returns the done status and reward for this bird for this timestep.
    std::pair<bool, double> get_done_reward(EnvAction & action, int agent){
        Flock & self = *this;
        double reward = 0;
        bool done = false;
        Bird & bird = self.birds[agent];

        /*
        If the bird has moved forward (in the x direction) in the last time step,
        it is rewarded.
        */
        if (len(bird.X) >= 2 && bird.x > bird.X[len(bird.X)-2]){
            reward += self.forward_reward;
        }

        /*
        Energy used by the bird is proportional to the birds thrust.
        */
        reward += self.energy_reward * action[0];

        //If the bird has crashed, we consider it done and punish it for crashing.
        if (self.crashed(bird)){
            done = true;
            reward = self.crash_reward;
        }

        //This can be changed depending on your goals.
        //Gives the bird a destination, flying 500m forward ends the simulation.
        if (bird.x > 500.0){
            done = true;
        }

        return std::make_pair(done, reward);
    }

    /*
    Checks to see if the bird has crashed in some way.
    */
    bool crashed(const Bird & bird){
        Flock & self = *this;

        //Checks if bird ahs hit the ground, or gone too high
        if (bird.z <= 0 || bird.z >= 100){
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
        extend(obs, force);
        extend(obs, torque);
        extend(obs, {bird.x, bird.y, bird.z});
        extend(obs, {bird.u, bird.v, bird.w});
        extend(obs, {bird.p, bird.q, bird.r});
        extend(obs, {bird.phi, bird.theta, bird.psi});
        extend(obs, {bird.alpha_l, bird.beta_l, bird.alpha_r, bird.beta_r});
        std::vector<Bird *> nearest = bird.n_nearest(self.birds, max_observable_birds);
        for (Bird * otherp : nearest){
	        Bird & other = *otherp;
            extend(obs, {other.x - bird.x, other.y - bird.y, other.z - bird.z});
            extend(obs, {other.u - bird.u, other.v - bird.v, other.w - bird.w});
            extend(obs, {other.phi - bird.psi, other.theta - bird.theta, other.psi - bird.psi});
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
        double limit_alpha = PI/6.0;
        double limit_beta = PI/4.0;

        /*
        Calculate the new alpha angle for each wing, taking the limits into account.
        The wing angles cannot exceed the maximums or minimums,
        and if an action tries to exceed these values, the wings will just
        remain at the max or min value.
        */
        double new_al = bird.alpha_l + action[1];
        if (new_al > limit_alpha)
            new_al = limit_alpha;
        if (new_al < -limit_alpha)
            new_al = -limit_alpha;
        bird.alpha_l = new_al;

        double new_bl = bird.beta_l + action[2];
        if (new_bl > limit_beta)
            new_bl = limit_beta;
        if (new_bl < -limit_beta)
            new_bl = -limit_beta;
        bird.beta_l = new_bl;

        double new_ar = bird.alpha_r + action[3];
        if (new_ar > limit_alpha)
            new_ar = limit_alpha;
        if (new_ar < -limit_alpha)
            new_ar = -limit_alpha;
        bird.alpha_r = new_ar;

        double new_br = bird.beta_r + action[4];
        if (new_br > limit_beta)
            new_br = limit_beta;
        if (new_br < -limit_beta)
            new_br = -limit_beta;
        bird.beta_r = new_br;
    }


    Birds get_birds(){
        return this->birds;
    }
};
