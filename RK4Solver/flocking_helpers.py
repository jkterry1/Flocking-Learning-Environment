import numpy as np

def update_bird(env, action):
    thrust = action[0]
    bird = env.birds[env.agent_selection]
    update_angles(env, action)
    vortices = get_vortices(env, bird)
    bird.update(thrust, env.h, vortices)

def get_reward(env, action):
    reward = 0
    done = False
    bird = env.birds[env.agent_selection]
    if bird.x > bird.X[-2]:
        reward += env.forward_reward
    reward -= env.energy_punishment * action[0]

    if crashed(env, bird):
        done = True
        reward = env.crash_reward

    if bird.x > 500.0:
        done = True

    return done, reward

def crashed(env, bird):
    if bird.z <= 0 or bird.z >= 100:
        return True

    lim = 2*np.pi
    if abs(bird.p) > lim or abs(bird.q) > lim or  abs(bird.r) > lim:
        return True

    crash = False
    for b in env.birds:
        other = env.birds[b]
        if other is not bird:
            dist = np.sqrt((bird.x - other.x)**2 + (bird.y - other.y)**2 + (bird.z - other.z)**2)
            if dist < bird.Xl/2.0:
                crash = True
    return crash

def get_vortices(env, curr):
    vortices = []
    for b in env.birds:
        bird = env.birds[b]
        if bird is not curr:
            for vorts in [bird.VORTICES_LEFT, bird.VORTICES_RIGHT]:
                i = 0
                v = vorts[i]

                #want first vortex ahead of current vortex
                while i < len(vorts) and v.x < curr.x:
                    v = vorts[i]
                    i = i+1
                if i < len(vorts) and v.x >= curr.x:
                    r = np.sqrt((curr.y - v.y)**2 + (curr.z - v.z)**2)
                    if r < env.max_r:
                        vortices.append(v)
    return vortices

def get_observation(env, agent):
    force = env.birds[agent].F
    torque = env.birds[agent].T

    bird = env.birds[agent]
    pos = []
    pos += [bird.x, bird.y, bird.z]
    pos += [bird.u, bird.v, bird.w]
    pos += [bird.p, bird.q, bird.r]
    pos += [bird.phi, bird.theta, bird.psi]
    pos += [bird.alpha_l, bird.beta_l, bird.alpha_r, bird.beta_r]
    nearest = bird.seven_nearest(env.birds)
    for other in nearest:
        pos += [other.x - bird.x, other.y - bird.y, other.z - bird.z]
        pos += [other.u - bird.u, other.v - bird.v, other.w - bird.w]

    obs = np.array(force + torque + pos)
    return obs

def update_vortices(env):
    for b in env.birds:
        bird = env.birds[b]
        bird.shed_vortices()

        if env.LIA:
            bird.update_vortex_positions(bird.VORTICES_RIGHT, env.h*10.0)
            bird.update_vortex_positions(bird.VORTICES_LEFT, env.h*10.0)

        #remove expired vortices
        if env.steps > 1.0/env.h:
            a = bird.VORTICES_LEFT.pop(0)
            b = bird.VORTICES_RIGHT.pop(0)
            env.total_vortices += 2.0
            env.total_dist += a.dist_travelled
            env.total_dist += b.dist_travelled

def update_angles(env, action):
    bird = env.birds[env.agent_selection]
    limit_alpha = np.pi/6.0
    limit_beta = np.pi/4.0
    new_al = bird.alpha_l + action[1]
    if new_al > limit_alpha:
        new_al = limit_alpha
    if new_al < -limit_alpha:
        new_al = -limit_alpha
    bird.alpha_l = new_al

    new_bl = bird.beta_l + action[2]
    if new_bl > limit_beta:
        new_bl = limit_beta
    if new_bl < -limit_beta:
        new_bl = -limit_beta
    bird.beta_l = new_bl

    new_ar = bird.alpha_r + action[3]
    if new_ar > limit_alpha:
        new_ar = limit_alpha
    if new_ar < -limit_alpha:
        new_ar = -limit_alpha
    bird.alpha_r = new_ar

    new_br = bird.beta_r + action[4]
    if new_br > limit_beta:
        new_br = limit_beta
    if new_br < -limit_beta:
        new_br = -limit_beta
    bird.beta_r = new_br
