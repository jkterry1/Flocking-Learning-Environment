#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
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
        .def("update_vortices",&Flock::update_vortices)
        .def("get_observation",[](Flock & arg, int agent) {
            Observation obs = arg.get_observation(agent);
            size_t size = obs.size();
            float *foo = new float[size];
            std::copy(obs.begin(),obs.end(), foo);

            // Create a Python object that will free the allocated
            // memory when destroyed:
            py::capsule free_when_done(foo, [](void *f) {
                float *foo = reinterpret_cast<float *>(f);
                delete[] foo;
            });

            return py::array_t<float>(
                {size}, // shape
                {sizeof(float)}, // C-style contiguous strides for double
                foo, // the data pointer
                free_when_done); // numpy array references this parent
        });
        //#.def_property("", &Pet::getName, &Pet::setName)
}
