import pettingzoo.tests.api_test as api_test
import pettingzoo.tests.bombardment_test as bombardment_test
import pettingzoo.tests.performance_benchmark as performance_benchmark

from pettingzoo.render_test import render_test as render_test
from pettingzoo.error_tests import error_test as error_test
from pettingzoo.seed_test import seed_test as seed_test
from pettingzoo.save_obs_test import test_save_obs as test_save_obs
from pettingzoo.max_cycles_test import max_cycles_test as max_cycles_test

import flocking_env


def test_flocking_env():
    env = flocking_env.env()
    api_test(env, num_cycles=1000, render=False, verbose_progress=True)
    seed_test(env, num_cycles=1000)
    performance_benchmark(env)

# test render (both render test and api test w/ render)
# test equivalent of save obs?
# error tests
