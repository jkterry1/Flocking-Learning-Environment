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
    double energy_punishment;
    double forward_reward;
    double crash_reward;
    Birds birds;
    BirdInits starting_conditions;
    Flock(  int N,
            double h,
            double t,
            double energy_punishment,
            double forward_reward,
            double crash_reward,
            BirdInits init_vals,
            bool LIA
         ){
        Flock & self = *this;
        self.h = h;
        self.t = t;
        self.N = N;
        self.LIA = LIA;

        self.max_r = 1.0;

        self.energy_punishment = energy_punishment;
        self.forward_reward = forward_reward;
        self.crash_reward = crash_reward;

        for (size_t i : range(N)){
            self.birds.emplace_back(init_vals[i]);
        }
        self.starting_conditions = init_vals;
    }

    void reset(){
        Flock & self = *this;
        for (size_t i : range(self.N)){
            self.birds[i] = Bird(self.starting_conditions[i]);
        }
    }

    void update_bird(EnvAction action, int agent){
        Flock & self = *this;
        double thrust = action[0];
        Bird & bird = self.birds[agent];
        self.update_angles(action, agent);
        Vorticies vortices = self.get_vortices(bird);
        bird.update(thrust, self.h, vortices);
    }

    std::pair<bool, double> get_done_reward(EnvAction & action, int agent){
        Flock & self = *this;
        double reward = 0;
        bool done = false;
        Bird & bird = self.birds[agent];
        if (len(bird.X) >= 2 && bird.x > bird.X[len(bird.X)-2]){
            reward += self.forward_reward;
        }
        reward += self.energy_punishment * action[0];

        if (self.crashed(bird)){
            done = true;
            reward = self.crash_reward;
        }

        if (bird.x > 500.0){
            done = true;
        }

        return std::make_pair(done, reward);
    }

    bool crashed(const Bird & bird){
        Flock & self = *this;
        if (bird.z <= 0 || bird.z >= 100){
            return true;
        }

        double lim = 2*PI;
        if (abs(bird.p) > lim || abs(bird.q) > lim ||  abs(bird.r) > lim){
            return true;
        }

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

    Vorticies get_vortices(Bird & curr){
        Flock & self = *this;
        Vorticies vortices;
        for (size_t b : range(len(self.birds))) {
            const Bird & bird = self.birds[b];
            if (&bird != &curr){
                for (const Vorticies & vorts : {bird.VORTICES_LEFT, bird.VORTICES_RIGHT}){
                    size_t i = 0;
                    //want first vortex ahead of current vortex
                    while (i < len(vorts) && vorts[i].x() < curr.x){
                        i = i+1;
                    }
                    const Vortex & v = vorts[i];
                    if (i < len(vorts) && v.x() >= curr.x){
                        // TODO: what about the x dimention? Don't you care about the radius in all 3 dimentions? Although I don't supos it matters that much
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
        }
        return obs;
    }

    void update_vortices(double vortex_update_frequency){
        Flock & self = *this;
        for (size_t b : range(len(self.birds))){
            Bird & bird = self.birds[b];
            bird.shed_vortices();

            if (self.LIA){
                bird.update_vortex_positions(bird.VORTICES_RIGHT, self.h*vortex_update_frequency);
                bird.update_vortex_positions(bird.VORTICES_LEFT, self.h*vortex_update_frequency);
            }

            //remove expired vortices
            if (len(bird.VORTICES_RIGHT) > 1.0/(self.h*vortex_update_frequency)){
                pop0(bird.VORTICES_LEFT);
                pop0(bird.VORTICES_RIGHT);
            }
        }
    }

    void update_angles(EnvAction action,int agent){
        Flock & self = *this;
        Bird & bird = self.birds[agent];
        double limit_alpha = PI/6.0;
        double limit_beta = PI/4.0;
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
