#include "Eigen/Dense"
#include "types.hpp"
#include "utils.hpp"

using namespace Eigen;

/*
  dr/dt describes the time evolution of the position of the vortex center.
*/
Vector3d drdt(double gamma, double epsilon, Vector3d b, double theta){
    return (gamma/(4.0* PI * epsilon)) * b * tan(theta/2.0);
}

/*
  d(u,v,w)/dt describes the time evolution of the velocity vector (u,v,w).
  This is equivalent to calculating the acceleration.
*/
Vector3d duvwdt(Vector3d uvw, Bird & bird){
    //F contains the force vector due to all other forces (drag, thrust, etc.)
    Vector3d F = Vector3d(Fu(uvw[0], bird), Fv(uvw[1], bird), Fw(uvw[2], bird));

    Vector3d grav = Vector3d(-bird.g * sin(bird.ang[0]),
                      bird.g * cos(bird.ang[0]) * sin(bird.ang[1]),
                      bird.g * cos(bird.ang[0]) * cos(bird.ang[1]));

    return (1.0 / bird.m) * F - cross(bird.pqr, uvw) + grav;
}

/*
  d(x, y, z)/dt describes the time evolution of the position vector (x, y, z)
  of the center of the bird.
  This is equivalent to calculatig the velocity.
*/
Vector3d dxyzdt(Vector3d xyz, Bird & bird){
    /*
    These three matricies are rotations that converts the velocity values
    from the bird's frame to the earth frame
    */
    double sphi = sin(bird.ang[0]);
    double cphi = cos(bird.ang[0]);
    double stheta = sin(bird.ang[1]);
    double ctheta = cos(bird.ang[1]);
    double spsi = sin(bird.ang[2]);
    double cpsi = cos(bird.ang[2]);

    Matrix3d mat = matrix(
            cphi * ctheta   , cpsi * stheta * sphi - spsi * cphi    , spsi * sphi + cpsi * cphi * stheta,
            spsi * ctheta   , cpsi * cphi + sphi * stheta * spsi    , stheta * spsi * cphi - cpsi * sphi,
            -stheta         , ctheta * sphi                         , ctheta * cphi
            );

    return matmul(mat, bird.uvw);
}

/*
  d(p,q,r)/dt describes the time evolution of the angular velocity (p, q, r)
  vector.
  This is equivalent to the angular acceleration.
*/
Vector3d dpqrdt(Vector3d pqr, Bird & bird){
    Vector3d LMN = Vector3d(TL(bird, pqr[0]), TM(bird, pqr[1]), TN(bird, pqr[2]));

    Matrix3d mat = matrix(0.0, -pqr[2], pqr[1],
                 pqr[2], 0.0, -pqr[0],
                 -pqr[1], pqr[0], 0.0);

    Vector3d rhs = LMN - matmul(mat, matmul(bird.inertia, pqr));

    /*
      Right hand side of the matrix equation that is used to
      calculate p-dot, q-dot, and r-dot.
      This needs to be inverted to find the final (p-dot, q-dot, and r-dot).
    */
    return matmul(bird.inertia.inverse(), rhs);
}

/*
  d(angles)/dt represents the time evolution of the euler angles of the bird,
  (phi, theta, psi).
  This is equivalent to calculating the angular velocity.
*/
Vector3d danglesdt(Vector3d ang, Bird & bird){
    Matrix3d mat = matrix(1.0, sin(ang[0])*tan(ang[1]), cos(ang[0])*tan(ang[1]),
                      0.0, cos(ang[0]), -sin(ang[0]),
                      0.0, sin(ang[0])*(1.0/cos(ang[1])), cos(ang[0])*(1.0/cos(ang[1])));
    Vector3d output = matmul(mat, bird.pqr);
    return output;
}

/*
  TL is the torque around the u axis.
  The contributions to the torque come from upward lift on each wing,
  drag on each wing due to rotation, and vortices if they are present.
*/
double TL(Bird & bird, double P){
    double T = 0;
    double v,A,r,Tl,Tr;
    double cl, cr, cd;

    cd = 1.5;

    //Finds the coefficient of lift for the bird.
    //This depends on angle of attack from the bird's wing orientation.
    up(cl, cr) = C_lift(bird);

    //torque due to upward lift on left wing
    //Lift = Cl * (rho * v^2)/2 * A
    //There is only upward lift if the bird is moving forward.
    if (bird.uvw[0] > 0){
        //velocity
        v = bird.uvw[0];
        //Approximate area of left wing
        A = bird.Xl * bird.Yl * cos(bird.alpha_l);
        //Verifies the area of the wing.
        check_A(bird, A, bird.alpha_r);
        //distance at which the force is applied
        r = bird.Xl/2.0;
        Tl = cos(bird.beta_l) * r * cl * A * (bird.rho * sqr(v))/2.0;
        T += Tl;


        //torque due to upward lift on right wing
        A = bird.Xl * bird.Yl * cos(bird.alpha_r);
        check_A(bird, A, bird.alpha_r);
        r = bird.Xl/2.0;
        Tr = -cos(bird.beta_r) * r * cr * A * (bird.rho * sqr(v))/2.0;
        T += Tr;

        //torque due to lift in v direction from left wing
        A = bird.Xl * bird.Yl * cos(bird.alpha_l);
        check_A(bird, A, bird.alpha_l);
        r = abs(sin(bird.beta_l)) * bird.Xl/2.0;
        Tl = abs(sin(bird.beta_l)) * r * cl * A * (bird.rho * sqr(v))/2.0;
        T += Tl;

        //torque due to lift in v direction from right wing
        A = bird.Xl * bird.Yl * cos(bird.alpha_r);
        check_A(bird, A, bird.alpha_r);
        r = abs(sin(bird.beta_r)) * bird.Xl/2.0;
        Tr = -abs(sin(bird.beta_r)) * r * cr * A * (bird.rho * sqr(v))/2.0;
        T += Tr;
    }

    //drag torque due to rotation on left wing
    //Td = cd * A * (rho * v^2)/2 * r
    //v = omega * r
    v = P * bird.Xl/2.0;
    r = bird.Xl/2.0;
    A = bird.Xl * bird.Yl;
    check_A(bird, A, 0.0);
    Tl = -sign(P) * r * A * cd * (bird.rho * sqr(v))/2.0;
    T += Tl;

    //drag due to rotation on right wing
    v = P * bird.Xl/2.0;
    r = bird.Xl/2.0;
    A = bird.Xl * bird.Yl;
    Tr = -sign(P) * r * A * cd * (bird.rho * sqr(v))/2.0;
    T += Tr;

    //cout<<"TL: "<<T<<" TL left wing: "<<Tl<<" right wing: "<<Tr<<"\n";

    //torque due to vortices, this is pre calculated
    T += bird.vortex_tuvw[0];
    bird.T[0] = T;
    return T;
}

/*
  TM is torque around the v axis (out of the right wing in the bird's frame)
  Forces that contribute to this torque are drag due to rotation and forces
  due to any present vortices.
*/
double TM(Bird & bird, double Q){
    double T = 0;
    //Distance at which the force acts.
    double r = bird.Yl/4.0;
    //The tangential velocity of the air is (omega * r),
    //where omega is the angular velocity.
    double v = (Q * r)/2.0;
    double A = (bird.Xl * bird.Yl);

    //Drag force due to rotation
    double D = -sign(v) * r * bird.Cd * A * (bird.rho * sqr(v))/2.0;

    //Both wings contribute D drag
    T += 2.0 * D;
    T += bird.vortex_tuvw[1];
    bird.T[1] = T;

    //cout<< "TM: "<<D<<" Q: "<<Q<<"\n";
    return T;
}

/*
  TN is the torque around the w axis (up through the center of the bird)
  Forces that contribute to this torque are drag on any wings the are
  rotated forward or backwards, drag due to rotation, and drag
  due to any present vortices.
*/
double TN(Bird & bird, double R){
    double T = 0;
    double v,r,A,Tdl,Tdr;

    //drag from left wing (this comes from any alpha rotation of the wing)
    v = bird.uvw[1];
    r = bird.Xl/2.0;
    //Area of the wing that is perpendicular to the forward motion.
    //If there is no alpha rotation, then there is no area of the wing
    //affected by drag.
    A = bird.Xl * abs(sin(bird.alpha_l)) * bird.Yl;
    check_A(bird, A, bird.alpha_l);
    //Torque due to drag on the left wing
    Tdl = sign(bird.uvw[0]) * r * bird.Cd * A * (bird.rho * sqr(v))/2.0;
    T += Tdl;

    //drag on right wing (this comes from any alpha rotation of the wing)
    v = bird.uvw[0];
    r = bird.Xl/2.0;
    //Area of the wing that is perpendicular to the forward motion.
    //If there is no alpha rotation, then there is no area of the wing
    //affected by drag.
    A = bird.Xl * abs(sin(bird.alpha_r)) * bird.Yl;
    check_A(bird, A, bird.alpha_r);
    //Torque due to drag on the right wing
    Tdr = -sign(bird.uvw[0]) * r * bird.Cd * A * (bird.rho * sqr(v))/2.0;
    T += Tdr;

    //drag from rotation on left wing
    //Tangential velocity is (angular velocity) * r
    v = R * bird.Xl/2.0;
    r = bird.Xl/2.0;
    //Area of the left wing that is perpendicular to the direction of rotation.
    A = bird.Xl * bird.Yl * abs(sin(bird.alpha_l)) + bird.Yl * bird.Zl * cos(bird.alpha_r);
    check_A(bird, A, bird.alpha_r);
    //Torque due to drag from rotation
    Tdl = -sign(bird.pqr[2]) * r * bird.Cd * (bird.rho * sqr(v))/2.0;
    T += Tdl;

    //drag from rotation on right wing
    //Tangential velocity is (angular velocity) * r
    v = R * bird.Xl/2.0;
    r = bird.Xl/2.0;
    //Area of the right wing that is perpendicular to the direction of rotation.
    A = bird.Xl * bird.Yl * abs(sin(bird.alpha_r)) + bird.Yl * bird.Zl * cos(bird.alpha_r);
    check_A(bird, A, bird.alpha_r);
    Tdr = -sign(bird.pqr[2]) * r * bird.Cd * (bird.rho * sqr(v))/2.0;
    T += Tdr;

    //torque due to drag from vortices
    T += bird.vortex_tuvw[2];

    bird.T[2] = T;
    return T;
}

/*
  Fu is the force on the bird in the u (through the nose of the bird) direction.
  Thrust, drag, and any nearby vortices contribute to this force.
*/
double Fu(double u, Bird & bird){
    double F = 0.0;
    double alpha_l = abs(bird.alpha_l);
    double alpha_r = abs(bird.alpha_r);

    //Area of the wing that is perpendicular to the forwad direction of motion.
    double A = bird.Xl * bird.Zl + bird.Xl * bird.Yl * sin(alpha_l) +\
        bird.Xl * bird.Yl * sin(alpha_r);
    double D = -sign(u) * bird.Cd * A * (bird.rho * sqr(u))/2.0;
    //bird.Fd[0] = D;

    F += D;
    //Force due to vortices. This is pre-calculated
    F += bird.vortex_fuvw[0];
    //Force from the thrust. Thrust is determined solely by the action taken.
    F += bird.thrust;

    bird.F[0] = F;
    return F;
}

/*
  Fv is the force on the bird in the v direction (out the right wing).
  Lift, drag, and vortices contribute to this force.
*/
double Fv(double v, Bird & bird){
    double F = 0;
    double A = bird.Xl * bird.Yl;
    double cl, cr;
    up(cl, cr) = C_lift(bird);

    //Lift on the left and right wongs
    double L_left = cl * A * (bird.rho * sqr(v))/2.0;
    double L_right = cr * A * (bird.rho * sqr(v))/2.0;

    A = bird.Yl * bird.Zl;
    //Drag on the bird
    double D = -sign(v) * bird.Cd * A * (bird.rho * sqr(v))/2.0;
    //bird.Fd[1] = D;

    //If the bird is moving forward, there is a lift contribution to the force
    //from each wing.
    if (bird.uvw[0] > 0.0){
        //cout<<"lift v \n";
        F += L_left * sin(bird.beta_l);
        F += -L_right * sin(bird.beta_r);
    }
    F += D;
    F += bird.vortex_fuvw[1];
    //cout<<"vortex force v: "<<bird.vortex_force_v<<'\n';
    bird.F[1] = F;
    return F;
}

/*
  Fw is the force on the bird in the w direction,
  (up through the center of the bird).
  Lift, drag, and vortices contribute to this force.
*/
double Fw(double w, Bird & bird){
    double F = 0;
    double A = bird.Xl * bird.Yl;
    double cl, cr;

    //Coefficient of lift for the left and right wing.
    //This depends on the orientaion (alpha, beta) of each wing
    //(which changes the angle of attack)
    up(cl, cr) = C_lift(bird);

    //Lift due to left and right wing
    double L_left = cl * A * (bird.rho * sqr(bird.uvw[0]))/2.0;
    double L_right = cr * A * (bird.rho * sqr(bird.uvw[0]))/2.0;

    //Drag force on the bird
    double D = -sign(w) * bird.Cd * A * (bird.rho * sqr(w))/2.0;
    //bird.Fd[2] = D;

    //There is only lift if the bird is flying forwards.
    if (bird.uvw[0] > 0.0){
        F += L_left * cos(bird.beta_l);
        F += L_right * cos(bird.beta_r);
    }
    F += D;
    F += bird.vortex_fuvw[2];
    bird.F[2] = F;
    return F;
}

/*
  Verifies that the bird is not broken and the area is nonzero.
*/
void check_A(Bird & bird, double A, double angle){
    if( A < 0){
        bird.broken = true;
    }
}

/*
  Calculates the coefficient of lift for a birds wings.
*/
Vector2d C_lift(Bird & bird){
    double d;
    if( bird.uvw[0] == 0){
        d = PI/2.0;
    }
    else{
        d = 0.0;
    }

    //Angle of attack on the left and right wings.
    double aoa_l = degrees(bird.alpha_l + d);
    double aoa_r = degrees(bird.alpha_r + d);

    //The maximum coefficient, this is a constant.
    // use reference instead to prevent copying the variable twice
    // and so we don't have to do bird.Cl_max everywhere
    const double &c_max = bird.Cl_max;
    double cl, cr;

    //If the angle of attack is too small or too large, there is no lift.
    if (aoa_l < -10 || aoa_l > 25){
        cl = 0;
    }

    //Under an angle of attack of 15, the coefficient is proportional to the aoa.
    else if (aoa_l < 15){
        cl = ((c_max/25.0) * aoa_l) + (10.0 * c_max / 25.0);
    }

    //An angle of attack under 20 but above 15 gives the max coefficient.
    else if (aoa_l < 20){
        cl = c_max;
    }

    //An angle of attack between 20 and 25 is proportional to -aoa
    else{
        cl = ((-c_max/25.0) * aoa_l) + (c_max + (20.0 * c_max / 25.0));
    }

    //Same rules as above for the right wing.
    if (aoa_r < -10 || aoa_r > 25){
        cr = 0;
    }
    else if( aoa_r < 15){
        cr = ((c_max/25.0) * aoa_r) + (10.0 * c_max / 25.0);
    }
    else if (aoa_r < 20){
        cr = c_max;
    }
    else{
        cr = ((-c_max/25.0) * aoa_r) + (c_max + (20.0 * c_max / 25.0));
    }

    return Vector2d(cl, cr);
}
