#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "flock.cpp"

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}
py::array_t<double> return_vec(Vector3d & vec){
    int size = 3;

    return py::array_t<double>(
        {size}, // shape
        {sizeof(double)}, // C-style contiguous strides for double
        &vec[0]); // numpy array references this parent
}

PYBIND11_MODULE(flocking_cpp, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");
    py::class_<Flock>(m, "Flock")
        .def(py::init<int,double,double,double,double,double,BirdInits,bool,bool,double,double,double,double>())
        .def("reset",&Flock::reset)
        .def("update_bird",&Flock::update_bird)
        .def("get_done_reward",&Flock::get_done_reward)
        .def("update_vortices",&Flock::update_vortices)
        .def("get_observation",[](Flock & arg, int agent, size_t max_observable_birds) {
            Observation obs = arg.get_observation(agent, max_observable_birds);
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
        })
        .def("get_bird",[](Flock & flock, int bird){
            return flock.birds[bird];
        })
        .def("get_birds",[](Flock & flock){return flock.birds;})
        ;

    py::class_<BirdInit>(m, "BirdInit")
        .def(py::init<double,double,double,double,double,double,double,double,double,double,double,double>())
        ;
    #define expose_bird_var(var) def_readonly(#var, &Bird::var)
    py::class_<Bird>(m, "Bird")
        .expose_bird_var(x)
        .expose_bird_var(y)
        .expose_bird_var(z)
        .expose_bird_var(u)
        .expose_bird_var(v)
        .expose_bird_var(w)
        .expose_bird_var(p)
        .expose_bird_var(q)
        .expose_bird_var(r)
        .expose_bird_var(theta)
        .expose_bird_var(phi)
        .expose_bird_var(psi)
        .expose_bird_var(alpha_l)
        .expose_bird_var(alpha_r)
        .expose_bird_var(beta_l)
        .expose_bird_var(beta_r)
        .expose_bird_var(thrust)
        .expose_bird_var(vortex_force_u)
        .expose_bird_var(vortex_force_v)
        .expose_bird_var(vortex_force_w)
        .expose_bird_var(vortex_torque_u)
        .expose_bird_var(vortex_torque_v)
        .expose_bird_var(vortex_torque_w)
        .def_property_readonly("F",[](Bird & bird){return return_vec(bird.F);})
        .def_property_readonly("T",[](Bird & bird){return return_vec(bird.T);})
        .expose_bird_var(U)
        .expose_bird_var(V)
        .expose_bird_var(W)
        .expose_bird_var(X)
        .expose_bird_var(Y)
        .expose_bird_var(Z)
        .expose_bird_var(P)
        .expose_bird_var(Q)
        .expose_bird_var(R)
        .expose_bird_var(THETA)
        .expose_bird_var(PHI)
        .expose_bird_var(PSI)
        .expose_bird_var(ALPHA_L)
        .expose_bird_var(ALPHA_R)
        .expose_bird_var(BETA_L)
        .expose_bird_var(BETA_R)
        .expose_bird_var(VORTICES_LEFT)
        .expose_bird_var(VORTICES_RIGHT)
        ;

    #define expose_vortex_var(var) def_readonly(#var, &Vortex::var)
    py::class_<Vortex>(m, "Vortex")
        .def(py::init<double, double, double, double, double, double, double>())
        .def_property_readonly("pos",[](Vortex & vortex){return return_vec(vortex.pos);})
        .expose_vortex_var(sign)
        .expose_vortex_var(C)
        .expose_vortex_var(min_vel)
        .expose_vortex_var(max_r)
        .expose_vortex_var(dist_travelled)
        .expose_vortex_var(theta)
        .expose_vortex_var(phi)
        .expose_vortex_var(psi)
        .expose_vortex_var(gamma)
        .expose_vortex_var(core)
        .expose_vortex_var(X)
        .expose_vortex_var(Y)
        .expose_vortex_var(Z)
        ;

}
