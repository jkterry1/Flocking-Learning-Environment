#pragma once
#include "Eigen/Dense"
Vector3d drdt(double gamma, double epsilon, Vector3d b, double theta);
Vector3d duvwdt(Vector3d uvw, Bird & bird);
Vector3d dxyzdt(Vector3d xyz, Bird & bird);
Vector3d dpqrdt(Vector3d pqr, Bird & bird);
Vector3d danglesdt(Vector3d angles, Bird & bird);
double TL(Bird & bird, double P);
double TM(Bird & bird, double Q);
double TN(Bird & bird, double R);
double Fu(double u, Bird & bird);
double Fv(double v, Bird & bird);
double Fw(double w, Bird & bird);
void check_A(Bird & bird, double A, double angle);
Vector2d C_lift(Bird & bird);
