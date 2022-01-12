import numpy as np
import fle.flocking_env as flocking_env

'''
this calculates the energy spent flying for 60s.
'''

t = 60.0
hz = 1000
h=1/hz
n = (int)(t/h)
LIA = False

distance_reward_per_m = 0
energy_reward_per_j = 1
crash_reward = 0

def run():
    #gliding - birds = [solver.make_bird(z=200.0, u=18.0)]
    u = 5.0
    z = 2000.0

    birds = [
            flocking_env.make_bird(y = 0.0, z=z, u = u),
            flocking_env.make_bird(y = 1.25, x = 0.5, z=z, u = u),
            flocking_env.make_bird(y = 2.5, z=z, u = u)]
    #birds = None
    #birds = [flocking_env.make_bird(y = 0.0, z=z, u = u)]

    env = flocking_env.raw_env(N = 3,
                                t=60.0,
                                h= 1/hz,
                                energy_reward=energy_reward_per_j,
                                forward_reward=distance_reward_per_m,
                                crash_reward=crash_reward,
                                bird_inits = birds,
                                LIA=True,
                                action_logging=True,
                                include_vortices=False)


    env.reset()
    done = False
    obs, reward, done, info = env.last()
    steps = 0

    energies = {i:0.0 for i in env.agents}
    dists = {i:0.0 for i in env.agents}
    agent = env.agents[0]

    for j in range(1):
        env.reset()
        dists[env.agent_selection] = 0
        for agent in env.agent_iter():
            obs, reward, done, info = env.last()
            dists[env.agent_selection] += obs[15] * h
            energies[env.agent_selection] += reward
            a = None
            if not done:
                w = obs[16]
                a = [0.0, 0.5, 0.5, 0.5, 0.5]
                if w < 0.5:
                    a = [1.0, 0.5, 0.5, 0.5, 0.5]
                a = np.array(a)
            env.step(a)
        energies[env.agent_selection] += reward
        dists[env.agent_selection] += obs[15] * h

    print("Energy spent in 60s in hardcoded V: ", energies[agent], "J")
    print("Distance travelled in 60s in hardcoded V: ", dists[agent], "m")

if __name__ == "__main__":
    run()
