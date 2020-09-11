#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "flock.cpp"

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");
    py::class_<Flock>(m, "Flock")
        .def(py::init<int,double,double,bool>())
        .def("reset",&Flock::reset)
        .def("update_bird",&Flock::update_bird)
        .def("get_reward",&Flock::get_reward)
        .def("get_observation",&Flock::get_observation)
        .def("update_vortices",&Flock::update_vortices);
        //#.def_property("", &Pet::getName, &Pet::setName)
}
