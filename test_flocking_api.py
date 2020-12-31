import pettingzoo.tests.api_test as api_test
import pettingzoo.tests.bombardment_test as bombardment_test
import pettingzoo.tests.performance_benchmark as performance_benchmark

from pettingzoo.tests import render_test
from pettingzoo.tests import error_test
from pettingzoo.tests import seed_test 
from pettingzoo.tests import test_save_obs
from pettingzoo.tests import max_cycles_test 

import flocking_env


env = flocking_env.env()
api_test.api_test(env, num_cycles=1000, render=False, verbose_progress=True)
seed_test.seed_test(env, num_cycles=1000)
performance_benchmark.performance_benchmark(env)

# test render (both render test and api test w/ render)
# test equivalent of save obs?
# error tests
