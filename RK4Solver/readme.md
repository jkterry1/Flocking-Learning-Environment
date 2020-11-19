## Run

install: `pip install pettingzoo pybind11`

compile: `bash build.sh`

run example: `python solver_tester`

NOTE: need to compile before running python code

### Code outline

Python:

* `flocking_env.py`: The PettingZoo environment.
* `plotting.py`: Plot utilities
* `solver_tester.py`: Tests bird flocking
* `test_flocking_api.py`: Tests the PettingZoo environment API

C++:

* `basic.cpp`: C++ testing code
* `bird.cpp`: Bird data structure, bird history
* `DiffEqs.cpp`: RK4 Differential equation solver
* `flock.cpp`: Environment level data structure
* `py_interface.cpp`: Python interface for environment
* `vortex.cpp`: Vortex data structure
