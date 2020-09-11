#include <iostream>
#include "Eigen/Dense"
#include <array>
#include <vector>

using namespace Eigen;
using namespace std;

struct Arg{
    static constexpr double arg = 10.2;
};
class unwrap3d{
public:
  double * xp,* yp,* zp;
  unwrap3d(double & x, double & y, double & z){
    xp = &x;
    yp = &y;
    zp = &z;
  }
  void operator =(Vector3d v){
    *xp = v(0);
    *yp = v(1);
    *zp = v(2);
  }
};

class unwrap2d{
public:
  double * xp,* yp;
  unwrap2d(double & x, double & y){
    xp = &x;
    yp = &y;
}
  void operator =(Vector2d v){
    *xp = v(0);
    *yp = v(1);
  }
};
Vector3d arr(std::array<double,3> a){
    Vector3d b;
    b(0) = a[0];
    return b;
}
unwrap3d p(double & x, double & y, double & z){
    return unwrap3d(x,y,z);
}
unwrap2d p(double & x, double & y){
    return unwrap2d(x,y);
}
inline Vector3d vector(std::array<double,3> vec){
    return Vector3d(vec[0],vec[1],vec[2]);
}
inline Matrix3d matrix(std::array<double,9> m){
    Matrix3d mat;
    mat << m[0],m[1],m[2],m[3],m[4],m[5],m[6],m[7],m[8];
    return mat;
}
void extend(std::vector<float> & obs, std::initializer_list<double> inits){
    for(double v : inits){
        obs.push_back(v);
    }
}
int main()
{
    Arg().arg;
  Matrix3d m = Matrix3d::Random();
  m = (m + Matrix3d::Constant(1.2)) * 50;
  cout << "m =" << endl << m << endl;
  Vector3d v(1,2,3);
  Vector2d v2(2,3);
  arr({1.,2.,5});
  matrix({1.,1.,1.,1.,1.,1.,1.,1.,1.});
  double x,y,z;
  Vector3d arg = dot(m,v);
  p(x,y,z) = v;
  std::vector<float> obs;
  extend(obs,{1.,3.,4.});
  double argv = v[0];

  cout << "m * v =" << endl << m * v << endl;
  cout << "x,y,z =" << endl << x + y + z << endl;
  cout << argv << endl;
}
