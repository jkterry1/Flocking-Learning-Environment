from pettingzoo.tests import api_test
from pettingzoo.tests import bombardment_test
from pettingzoo.tests import performance_benchmark
from pettingzoo.tests import render_test
from pettingzoo.tests import error_test
from pettingzoo.tests import seed_test
from pettingzoo.tests import test_save_obs
from pettingzoo.tests import max_cycles_test

import flocking_env

env = flocking_env.env()
api_test.api_test(env, num_cycles=1, render=False, verbose_progress=True)
performance_benchmark.performance_benchmark(env)

# test render (both render test and api test w/ render)
# test equivalent of save obs?
# error tests?
# seed tets
