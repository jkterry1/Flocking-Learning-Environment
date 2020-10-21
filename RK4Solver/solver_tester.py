import numpy as np
#import solver_env as solver
import flocking_env as solver
import time


t = 10.0
h = 0.001
n = (int)(t/h)
N = 10
LIA = False

def run():
    tik = time.time()
    z = 0.01
    birds = [solver.make_bird(z=90.0, y=0.6, x=-1.0, u=0, p = 0)]
    env = solver.env(N = N, LIA = LIA)
    #env = solver.env(N = 10, LIA = LIA)
    # env.flock.get_bird(1).x=1.
    # print(env.flock.get_bird(1).x)
    # print(setattr(env.flock.get_bird(1),"x",1.))
    # print(getattr(env.flock.get_bird(1),"x"))
    env.reset()
    done = False
    start = time.time()
    for i in range(n):
        if i % 100 == 0:
            end = time.time()
            print(i, end-start)
            start = end
        if not done:
            for i in range(N):
                if not done:
                    #print(env.flock.get_bird(1).U)
                    #env.birds[agent].u = 7.1271
                    #env.birds[agent].print_bird(env.birds[agent])
                    a = np.zeros(5)
                    a[0] = 1
                    # if env.birds[agent].u < 7.1271:
                    #     a[0] = 2.0
                    obs = env.step(a)
                    #print(np.array(obs,dtype=np.float32))
                    done = env.dones[env.agent_selection]
                    rew = env.rewards[env.agent_selection]
                    #reward, done, info = env.last()
                    #print("bird ", i, " ", obs)
                else:
                    break

        else:
            break
            env.reset()
            done = False
            print("reset")
    print("timesteps ", i)
    tok = time.time()
    print("Time: ", tok - tik)
    print("Steps: ", n)
    print("Time per step: ", (tok-tik)/n)
    print("Steps per second: ", n/(tok-tik))

    env.render(plot_vortices = False)

if __name__ == "__main__":
    run()
