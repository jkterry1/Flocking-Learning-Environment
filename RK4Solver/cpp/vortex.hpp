#pragma once
#include <utility>
#include "Eigen/Dense"
using namespace Eigen;
class Bird;
struct Vortex{
    double sign;
    double C;
    double min_vel;
    double max_r;
    double dist_travelled;
    double theta;
    double phi;
    double psi;
    Vector3d pos;
    double gamma;
    double core;
    double x()const{return pos[0];}
    double y()const{return pos[1];}
    double z()const{return pos[2];}
  Vortex(Bird & bird, double sign);
  Vortex()=default;
    // vlocity of the vortex in the earth's frame
  Vector3d earth_vel(double x, double y, double z);

    // velocity in the bird's frame
  std::pair<Vector3d,Vector3d> bird_vel(Bird & bird);
  Matrix3d get_transform(double phi,double theta,double psi);
};
