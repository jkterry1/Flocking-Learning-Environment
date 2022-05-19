#pragma once
#include <utility>
#include "Eigen/Dense"
using namespace Eigen;
class Bird;

typedef Vector3d (DiffEqType)(Vector3d abc, Bird & bird);

struct Vortex{
    double sign;
    double C;
    double min_vel;
    double max_r;
    double dist_travelled;
    Vector3d ang;
    Vector3d pos;
    double gamma;
    double gamma_0;
    double decay_std_dev;
    double decay_time_mean;
    double decay_exponent;
    double t_decay;
    bool decaying;
    double t;
    double core;
    double x()const{return pos[0];}
    double y()const{return pos[1];}
    double z()const{return pos[2];}
    double X;
    double Y;
    double Z;
  Vortex(Bird & bird, double sign);
  Vortex(double x, double y, double z, double phi, double theta, double psi, double sign);
  Vortex()=default;
    // vlocity of the vortex in the earth's frame
  Vector3d earth_vel(double x, double y, double z);

    // velocity in the bird's frame
  std::pair<Vector3d,Vector3d> bird_vel(Bird & bird);
  /* Matrix3d get_transform(double phi,double theta,double psi); */
  Matrix3d get_transform(Vector3d ang);
};
