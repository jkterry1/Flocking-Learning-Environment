#!/bin/bash
c++ -O1 -g -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` cpp/py_interface.cpp -o flocking_cpp`python3-config --extension-suffix`
