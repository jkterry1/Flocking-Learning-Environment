import solver_env as solver
from baselines import deepq
from bird import Bird


def main():
    t = 5.0
    h = 0.1
    n = (int)(t/h)
    N = 1

    birds = [Bird(z = 20.0, x = 1.0, u = 5.0 , p = 0.0)]
    env = solver.raw_env(N = len(birds), h = h, birds = birds)
    act = deepq.learn(env = env, network = 'mlp')

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
