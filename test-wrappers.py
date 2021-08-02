import flocking_env
import supersuit as ss

n_evaluations = 20
n_agents = 9
n_envs = 4
total_energy_j = 46000
total_distance_m = 870
hz = 500
crash_reward = -10
episodes = 300
nerve_impulse_hz = 100
reaction_frames = 0

n_timesteps = hz*60*n_agents*episodes
distance_reward_per_m = 100/total_distance_m
energy_reward_per_j = -10/total_energy_j
skip_frames = int(hz/nerve_impulse_hz)
print("skip_frames: " + str(skip_frames))

def get_env():
    return flocking_env.env(N=n_agents, h=1/hz, energy_reward=energy_reward_per_j, forward_reward=distance_reward_per_m, crash_reward=crash_reward, LIA=True)

def test_delay(delay = 2, n = 10):
    env1 = get_env()
    env2 = get_env()
    env2 = ss.delay_observations_v0(env2, delay)

    env1.reset()
    env2.reset()

    obs1 = []
    obs2 = []

    a = [0.0, 0.5, 0.5, 0.5, 0.5]

    i=1
    for agent in env1.agent_iter():
        obs, reward, done, info = env1.last()
        obs1.append(obs)
        env1.step(a)
        i += 1
        if i >= n:
            break

    i=0
    for agent in env2.agent_iter():
        obs, reward, done, info = env2.last()
        obs2.append(obs)
        env2.step(a)
        i += 1
        if i >= n:
            break

    print(obs1)
    print(obs2)
    for i in range(delay, n):
        assert obs2[i] == obs1[i-delay]

test_delay()
'''
env = ss.frame_skip_v0(env, skip_frames)
env = ss.pettingzoo_env_to_vec_env_v0(env)
env = ss.concat_vec_envs_v0(env, n_envs, num_cpus=1, base_class='stable_baselines3')
env = VecMonitor(env)
'''
