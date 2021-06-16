#pragma once
#include "Eigen/Dense"
#include <cmath>
#include <ctime>

using namespace Eigen;

template<typename Type>
struct RangeIncr{
    struct RangeIteratorForward{
        Type x;
        bool operator != (RangeIteratorForward & Other){
            return x < Other.x;
        }
        void operator ++ (){
            x++;
        }
        Type & operator *(){
            return x;
        }
    };
    RangeIteratorForward Start;
    RangeIteratorForward End;
    RangeIncr(Type InStart, Type InEnd){
        Start = RangeIteratorForward{ InStart };
        End = RangeIteratorForward{InEnd};
    }
    RangeIteratorForward & begin(){
        return Start;
    }
    RangeIteratorForward & end(){
        return End;
    }
};
inline RangeIncr<size_t> range(size_t start,size_t end){
    return RangeIncr<size_t>(start,end);
}
inline RangeIncr<size_t> range(size_t end){
    return RangeIncr<size_t>(0,end);
}
template<class collection_ty>
size_t len(const collection_ty & data){
    return data.size();
}
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
unwrap3d up(double & x, double & y, double & z){
    return unwrap3d(x,y,z);
}
unwrap2d up(double & x, double & y){
    return unwrap2d(x,y);
}
template <typename T> int sign(T val) {
    return (T(0) < val) - (val < T(0));
}
template <typename T>
T degrees(T radians){
    return radians * (180.0/PI);
}
double dot(Vector3d v1, Vector3d v2){
    return v1.dot(v2);
}
Vector3d cross(Vector3d v1, Vector3d v2){
    return v1.cross(v2);
}
Vector3d abs(Vector3d v){
    return v.cwiseAbs();
}
Matrix3d matmul(Matrix3d m1, Matrix3d m2){
    return m1 * m2;
}
Vector3d matmul(Matrix3d m1, Vector3d m2){
    return m1 * m2;
}
double arccos(double x){
    return acos(x);
}
double arcsin(double x){
    return asin(x);
}
double arctan(double x){
    return atan(x);
}
template <typename T>
T sqr(T x){
    return x * x;
}
template <typename T>
void pop0(T & vec){
    vec.erase(vec.begin());
}
double time(){
    return clock()/double(CLOCKS_PER_SEC);
}


inline Vector3d vector(std::array<double,3> vec){
    return Vector3d(vec[0],vec[1],vec[2]);
}
inline Matrix3d matrix(
    double m0,
    double m1,
    double m2,
    double m3,
    double m4,
    double m5,
    double m6,
    double m7,
    double m8
){
    Matrix3d mat;
    mat << m0,m1,m2,m3,m4,m5,m6,m7,m8;
    return mat;
}

inline void extend(Observation & obs, std::initializer_list<double> inits){
    for(double v : inits){
        obs.push_back(v);
    }
}
inline void extend(Observation & obs, Vector3d vec){
    for(size_t i = 0; i < 3; i++){
        obs.push_back(vec[i]);
    }
}

inline Vector3d operator + (const Vector3d & vec, double d){
      return Vector3d{vec.x()+d,vec.y()+d,vec.z()+d};
}

inline Vector3d operator + ( double d, const Vector3d & vec){
      return Vector3d{vec.x()+d,vec.y()+d,vec.z()+d};
}
