import sys
from pettingzoo.test.api_test import api_test
from pettingzoo.test.performance_benchmark import performance_benchmark
from pettingzoo.test.manual_control_test import manual_control_test
from pettingzoo.test.render_test import render_test
from pettingzoo.test.seed_test import seed_test
from pettingzoo.test.save_obs_test import test_save_obs
from pettingzoo.test.max_cycles_test import max_cycles_test
from pettingzoo.test.parallel_test import parallel_api_test

import flocking_env


def perform_ci_test(num_cycles, render=False, performance=False):
    _env = flocking_env.env()
    error_collected = []

    api_test(_env, num_cycles=num_cycles, verbose_progress=True)
    parallel_api_test(flocking_env.parallel_env(), num_cycles=num_cycles)
    seed_test(flocking_env.env, num_cycles)

    if performance:
        _env = env_module.env()
        performance_benchmark(_env)

    return error_collected

print(perform_ci_test(10))
