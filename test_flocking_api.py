from pettingzoo.test import api_test
from pettingzoo.test import bombardment_test
from pettingzoo.test import performance_benchmark
from pettingzoo.test import render_test
from pettingzoo.test import seed_test
from pettingzoo.test import test_save_obs
from pettingzoo.test import max_cycles_test

import flocking_env

env = flocking_env.env()
api_test(env, num_cycles=1, render=False, verbose_progress=True)
performance_benchmark(env)
