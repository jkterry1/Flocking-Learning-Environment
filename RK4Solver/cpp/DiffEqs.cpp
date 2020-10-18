#include "Eigen/Dense"
#include "types.hpp"
#include "utils.hpp"

using namespace Eigen;


Vector3d drdt(double gamma, double epsilon, Vector3d b, double theta){
    return (gamma/(4.0* PI * epsilon)) * b * tan(theta/2.0);
}

Vector3d duvwdt(Vector3d uvw, Bird & bird){
    double theta = bird.theta;
    double phi = bird.phi;
    double psi = bird.psi;
    Vector3d pqr = bird.pqr();
    double u, v, w;
    up(u, v, w) = uvw;
    double m = bird.m;
    double g = bird.g;
    Vector3d grav = Vector3d(-g * sin(bird.theta),
		  g * cos(bird.theta) * sin(bird.phi),
		  g * cos(bird.theta) * cos(bird.phi));

    //print("grav force: ", grav)

    Vector3d F = Vector3d(Fu(u, bird), Fv(v, bird), Fw(w, bird));
    return (1.0/m)*F - cross(pqr, uvw) + grav;
}

Vector3d dxyzdt(Vector3d xyz, Bird & bird){
    double theta = bird.theta;
    double phi = bird.phi;
    double psi = bird.psi;
    Matrix3d R3 = matrix(cos(psi), -sin(psi), 0.0,
			 sin(psi), cos(psi), 0.0,
			 0.0, 0.0, 1.0);
    Matrix3d R2 = matrix(cos(theta), 0.0, sin(theta),
			 0.0, 1.0, 0.0,
			 -sin(theta), 0.0, cos(theta));
    Matrix3d R1 = matrix(1.0, 0.0, 0.0,
			 0.0, cos(phi), -sin(phi),
			 0.0, sin(phi), cos(phi));
    Vector3d output = matmul(R3, matmul(R2, matmul(R1, bird.uvw())));
    return output;
}


Vector3d dpqrdt(Vector3d pqr, Bird & bird){
    double p, q, r;
    up(p, q, r) = pqr;
    Vector3d LMN = Vector3d(TL(bird, p), TM(bird, q), TN(bird, r));
    Matrix3d inertia = bird.inertia;
    Matrix3d mat = matrix(0.0, -r, q,
			  r, 0.0, -p,
			  -q, p, 0.0);
    Vector3d rhs = LMN - matmul(mat, matmul(inertia, pqr));
    return matmul(inertia.inverse(), rhs);
}


Vector3d danglesdt(Vector3d angles, Bird & bird){
    double phi, theta, psi;
    up(phi, theta, psi) = angles;
    Vector3d pqr = bird.pqr();
    Matrix3d mat = matrix(1.0, sin(phi)*tan(theta), cos(phi)*tan(theta),
			  0.0, cos(phi), -sin(phi),
			  0.0, sin(phi)*(1.0/cos(theta)), cos(phi)*(1.0/cos(theta)));
    Vector3d output = matmul(mat, pqr);
    return output;
}

double TL(Bird & bird, double P){
    double T = 0;
    double v,A,r,Tl,Tr;
    double cl, cr;
    up(cl, cr) = C_lift(bird);
    //torque due to upward lift on left wing
    //Lift = Cl * (rho * v^2)/2 * A
    if (bird.u > 0){
        v = bird.u;
        A = bird.Xl * bird.Yl * cos(bird.alpha_l);
        check_A(bird, A, bird.alpha_r);
        r = bird.Xl/2.0;
        Tl = cos(bird.beta_l) * r * cl * A * (bird.rho * sqr(v))/2.0;
        //assert Tl >= 0

        T += Tl;
    }

    //torque due to upward lift on right wing
    if (bird.u > 0){ // TODO: are all these if statements supposed to be the same? That seems wrong
        v = bird.u;
        A = bird.Xl * bird.Yl * cos(bird.alpha_r);
        check_A(bird, A, bird.alpha_r);
        r = bird.Xl/2.0;
        Tr = -cos(bird.beta_r) * r * cr * A * (bird.rho * sqr(v))/2.0;
        //assert Tr <= 0

        T += Tr;
    }

    //torque due to lift in v direction from left wing
    if (bird.u > 0){
        v = bird.u;
        A = bird.Xl * bird.Yl * cos(bird.alpha_l);
        check_A(bird, A, bird.alpha_l);
        r = abs(sin(bird.beta_l)) * bird.Xl/2.0;
        Tl = abs(sin(bird.beta_l)) * r * cl * A * (bird.rho * sqr(v))/2.0;
        //assert Tl >= 0

        T += Tl;
    }

    //torque due to lift in v direction from right wing
    if (bird.u > 0){
        v = bird.u;
        A = bird.Xl * bird.Yl * cos(bird.alpha_r);
        check_A(bird, A, bird.alpha_r);
        r = abs(sin(bird.beta_r)) * bird.Xl/2.0;
        Tr = -abs(sin(bird.beta_r)) * r * cr * A * (bird.rho * sqr(v))/2.0;
        //assert Tr <= 0
        //assert sign(Tl) == 0 || sign(Tl) != sign(Tr)

        T += Tr;
    }

    //drag torque due to rotation on left wing
    //Td = cd * A * (rho * v^2)/2 * r
    //v = wr
    v = P * bird.Xl/2.0;
    r = bird.Xl/2.0;
    A = bird.Xl * bird.Yl;
    check_A(bird, A, 0.0);
    Tl = -sign(bird.p) * r * A * cl * (bird.rho * sqr(v))/2.0;
    //assert sign(Tl) == 0 || sign(Tl) != sign(bird.p)

    T += Tl;

    //drag due to rotation on right wing
    v = P * bird.Xl/2.0;
    r = bird.Xl/2.0;
    A = bird.Xl * bird.Yl;
    //assert A >= 0
    Tr = -sign(bird.p) * r * A * cr* (bird.rho * sqr(v))/2.0;
    //assert sign(Tr) == 0 || sign(Tr) != sign(bird.p)

    T += Tr;

    //torque due to vortices
    T += bird.vortex_torque_u;

    bird.T[0] = T;
    return T;
}

double TM(Bird & bird, double Q){
    double T = 0;
    double r = bird.Yl/4.0;
    double v = (Q * r)/2.0;
    double A = (bird.Xl * bird.Yl);
    //rotation drag
    double D = -sign(v) * r * bird.Cd * A * (bird.rho * sqr(v))/2.0;
    T += 2.0 * D;
    T += bird.vortex_torque_v;
    bird.T[1] = T;
    return T;
}

double TN(Bird & bird, double R){
    double T = 0;
    double v,r,A,Tdl,Tdr;
    //drag from left wing (including alpha rotation)
    //Td = cd * A * (rho * v^2)/2 * r
    v = bird.u;
    r = bird.Xl/2.0;
    A = bird.Xl * abs(sin(bird.alpha_l)) * bird.Yl;
    check_A(bird, A, bird.alpha_l);
    Tdl = sign(bird.u) * r * bird.Cd * A * (bird.rho * sqr(v))/2.0;

    T += Tdl;

    //drag on right wing (including alpha rotation)
    //Td = cd * A * (rho * v^2)/2 * r
    v = bird.u;
    r = bird.Xl/2.0;
    A = bird.Xl * abs(sin(bird.alpha_r)) * bird.Yl;
    check_A(bird, A, bird.alpha_r);
    Tdr = -sign(bird.u) * r * bird.Cd * A * (bird.rho * sqr(v))/2.0;

    T += Tdr;

    //drag from rotation on left wing
    //Td = cd * A * (rho * v^2)/2 * r
    //v = wr
    v = R * bird.Xl/2.0;
    r = bird.Xl/2.0;
    A = bird.Xl * bird.Yl * abs(sin(bird.alpha_l)) + bird.Yl * bird.Zl * cos(bird.alpha_r);
    check_A(bird, A, bird.alpha_r);
    Tdl = -sign(bird.r) * r * bird.Cd * (bird.rho * sqr(v))/2.0;
    //assert sign(Tdl) == 0 || sign(Tdl) != sign(bird.r)

    T += Tdl;

    //drag from rotation on right wing
    //Td = cd * A * (rho * v^2)/2 * r
    //v = wr
    v = R * bird.Xl/2.0;
    r = bird.Xl/2.0;
    A = bird.Xl * bird.Yl * abs(sin(bird.alpha_r)) + bird.Yl * bird.Zl * cos(bird.alpha_r);
    check_A(bird, A, bird.alpha_r);
    Tdr = -sign(bird.r) * r * bird.Cd * (bird.rho * sqr(v))/2.0;
    //assert sign(Tdr) == 0 || sign(Tdl) != sign(bird.r)

    T += Tdr;

    //drag from vortices
    T += bird.vortex_torque_w;

    bird.T[2] = T;
    return T;
}

double Fu(double u, Bird & bird){
    double F = 0.0;
    double alpha_l = abs(bird.alpha_l);
    double alpha_r = abs(bird.alpha_r);

    //Drag = Cd * (rho * v^2)/2 * A
    double A = bird.Xl * bird.Zl + bird.Xl * bird.Yl * sin(alpha_l) +\
        bird.Xl * bird.Yl * sin(alpha_r);
    //print("u before drag: ", u)
    double D = -sign(u) * bird.Cd * A * (bird.rho * sqr(u))/2.0;
    //print("Drag: ", D)

    F += D;
    F += bird.vortex_force_u;
    //print("Vortex force: ", bird.vortex_force_u)
    F += bird.thrust;
    //print("Thrust: ", bird.thrust)
    bird.F[0] = F;
    return F;
}

double Fv(double v, Bird & bird){
    double F = 0;
    //lift
    double A = bird.Xl * bird.Yl;
    double cl, cr;
    up(cl, cr) = C_lift(bird);
    double L_left = cl * A * (bird.rho * sqr(v))/2.0;
    double L_right = cr * A * (bird.rho * sqr(v))/2.0;
    //Drag = Cd * (rho * v^2)/2 * A
    A = bird.Yl * bird.Zl;
    double D = -sign(v) * bird.Cd * A * (bird.rho * sqr(v))/2.0;
    if (bird.u > 0.0){
        F += L_left * sin(bird.beta_l);
        F += -L_right * sin(bird.beta_r);
    }
    F += D;
    F += bird.vortex_force_v;
    bird.F[1] = F;
    return F;
}

double Fw(double w, Bird & bird){
    double F = 0;
    //Lift = Cl * (rho * v^2)/2 * S
    //Drag = Cd * (rho * v^2)/2 * A
    double A = bird.Xl * bird.Yl;
    double cl, cr;
    up(cl, cr) = C_lift(bird);
    double L_left = cl * A * (bird.rho * sqr(bird.u))/2.0;
    double L_right = cr * A * (bird.rho * sqr(bird.u))/2.0;
    double D = -sign(w) * bird.Cd * A * (bird.rho * sqr(w))/2.0;
    if (bird.u > 0.0){
        //left wing lift
        F += L_left * cos(bird.beta_l);
        //right wing lift
        F += L_right * cos(bird.beta_r);
    }
    F += D;
    F += bird.vortex_force_w;
    bird.F[2] = F;
    return F;
}

void check_A(Bird & bird, double A, double angle){
    if( A < 0){
        //bird.print_bird(bird);
        bird.broken = true;
    }
}


Vector2d C_lift(Bird & bird){
    double d;
    if( bird.u == 0){
        d = PI/2.0;
    }
    else{
        d = arctan(bird.w/bird.u);
    }
    double aoa_l = degrees(bird.alpha_l + d);
    double aoa_r = degrees(bird.alpha_r + d);
    double c_max = bird.Cl_max;
    double cl, cr;
    if (aoa_l < -10 || aoa_l > 25){
        cl = 0;
    }
    else if (aoa_l < 15){
        cl = ((c_max/25.0) * aoa_l) + (10.0 * c_max / 25.0);
    }
    else if (aoa_l < 20){
        cl = c_max;
    }
    else{
        cl = ((-c_max/25.0) * aoa_l) + (c_max + (20.0 * c_max / 25.0));
    }

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
