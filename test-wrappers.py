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
nerve_impulse_hz = 200
reaction_frames = 0

n_timesteps = hz*60*n_agents*episodes
distance_reward_per_m = 100/total_distance_m
energy_reward_per_j = -10/total_energy_j
skip_frames = int(hz/nerve_impulse_hz)

def get_env():
    return flocking_env.env(N=n_agents, h=1/hz, energy_reward=energy_reward_per_j, forward_reward=distance_reward_per_m, crash_reward=crash_reward, LIA=True)

def test_delay(delay = 4, n = 100):
    env1 = get_env()
    env2 = get_env()
    env2 = ss.delay_observations_v0(env2, delay)

    env1.reset()
    env2.reset()

    obs1 = []
    obs2 = []

    a = [0.0, 0.5, 0.5, 0.5, 0.5]

    i=0
    for agent in env1.agent_iter():
        obs, reward, done, info = env1.last()
        obs1.append((agent,obs))
        env1.step(a)
        i += 1
        if i >= n:
            break

    i=0
    for agent in env2.agent_iter():
        obs, reward, done, info = env2.last()
        obs2.append((agent,obs))
        env2.step(a)
        i += 1
        if i >= n:
            break

    # for i in range(len(obs1)):
    #     print(obs1[i][0],obs1[i][1][0])
    # print()
    # for i in range(len(obs2)):
    #     print(obs2[i][0],obs2[i][1][0])
    # print()

    for i in range(delay*n_agents,len(obs2)):
        #print((i, i-(delay*n_agents)))
        for j in range(len(obs2[i])):
            assert obs2[i][1][j] == obs1[i-(delay*n_agents)][1][j]

def test_frameskip(skip_frames = skip_frames, n = 100):
    env1 = get_env()
    env2 = get_env()
    env2 = ss.frame_skip_v0(env2, skip_frames)

    env1.reset()
    env2.reset()

    obs1 = []
    obs2 = []

    a = [0.0, 0.5, 0.5, 0.5, 0.5]

    i=0
    for agent in env1.agent_iter():
        obs, reward, done, info = env1.last()
        obs1.append((agent, obs))
        env1.step(a)
        i += 1
        if i >= n:
            break

    i=0
    for agent in env2.agent_iter():
        obs, reward, done, info = env2.last()
        obs2.append((agent, obs))
        env2.step(a)
        i += skip_frames
        if i >= n:
            break

    # for i in range(len(obs1)):
    #     print(obs1[i][0],obs1[i][1][0])
    # print()
    # for i in range(len(obs2)):
    #     print(obs2[i][0],obs2[i][1][0])
    # print()

    i2 = 0
    for i in range(len(obs2)):
        if i>0 and i%n_agents == 0:
            i2 += (skip_frames-1) * n_agents
        #print(i,i2)
        for j in range(len(obs2[i][1])):
            assert obs2[i][1][j] == obs1[i2][1][j]
        i2 += 1


test_delay()
test_frameskip()

'''
env = ss.frame_skip_v0(env, skip_frames)
env = ss.pettingzoo_env_to_vec_env_v0(env)
env = ss.concat_vec_envs_v0(env, n_envs, num_cpus=1, base_class='stable_baselines3')
env = VecMonitor(env)
'''
