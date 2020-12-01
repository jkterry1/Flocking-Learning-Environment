from pettingzoo.tests.api_test import api_test
from pettingzoo.tests.seed_test import seed_test
import flocking_env

def test_flocking_env():
    env = flocking_env.env()
    api_test(env)#, num_cycles=100)
    seed_test(flocking_env.env)

if __name__ == "__main__":
    test_flocking_env()