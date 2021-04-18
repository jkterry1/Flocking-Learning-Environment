#include "vortex.hpp"

Vortex::Vortex(double x, double y, double z, double phi, double theta, double psi, double sign){
    Vortex & self = *this;
    self.sign = sign;
    self.C = 15.0;
    self.min_vel = 1.0;
    self.max_r = 1.0;
    self.dist_travelled = 0.0;

    self.phi = phi;
    self.theta = theta;
    self.psi = psi;

    //mat transforms vectors from the bird's frame to the earth's frame.
    Matrix3d mat = self.get_transform(self.phi, self.theta, self.psi);

    self.pos = Vector3d(x,y,z);
    self.X = pos[0];
    self.Y = pos[1];
    self.Z = pos[2];

    //velocity of the vortex
    double vel = 0.0;

    self.gamma_0 = 5.0;
    self.gamma = self.gamma_0;
    self.decay_std_dev = 10.0;
    self.decay_time_mean = 3.0 * self.decay_std_dev;
    self.decay_exponent = 2.0;
    self.t_decay = 0.0;
    self.t = 0.0;

    //core represents the radius of the vortex's "core", the center of the
    //vortex where no air is moving to prevent divide by zero issue.
    self.core = 0.05;
}

Vortex::Vortex(Bird & bird, double sign){
    Vortex & self = *this;
    self.sign = sign;
    self.C = 15.0;
    self.min_vel = 1.0;
    self.max_r = 1.0;
    self.dist_travelled = 0.0;

    /*
      orientation
      -phi represents the angle rotated around the u axis,
        the axis that points out through the birds nose.
      -theta represents the angle rotated around the v axis,
        the axis that points out the right wing of the bird.
      -psi represents the angle rotated around the w axis,
        the axis that points up through the center of the bird.
    */
    self.phi = bird.phi;
    self.theta = bird.theta;
    self.psi = bird.psi;

    /*
      The angle of the vortex depends on the angle of the bird that produces
      it, and also the angle of the bird's wing.
    */
    if (sign == 1) // right wing
       self.phi += bird.beta_r;
    else //left wing
       self.phi -= bird.beta_l;

    //mat transforms vectors from the bird's frame to the earth's frame.
    Matrix3d mat = self.get_transform(self.phi, self.theta, self.psi);

    /*
      We need to transform the position of the vortex produced from the bird's
      frame into the earth's frame.
      First we find where the vortex is dropped, from the left or right wing,
      then use the frame change transformation.
    */
    //end_of_wing is the coordinate of the right wing tip in the bird's frame.
    Vector3d end_of_wing = Vector3d(0.0, bird.Xl, 0.0);
    if (sign == 1) // right wing
	     self.pos = bird.xyz() -  matmul(mat, end_of_wing);
    else //left wing
	     self.pos = bird.xyz() +  matmul(mat, end_of_wing);

    //velocity of the vortex
    double vel = bird.uvw().norm();
    /*
      If the bird is moving too slowly, it's will not produce a vortex.
      If it does produce a vortex, it's strength is inversely proportional
      to the bird's velocity.
    */
    if (abs(vel) > self.min_vel)
	   self.gamma = self.C * bird.m/(bird.Xl * vel);
    else
	   self.gamma = 0.0;

    //core represents the radius of the vortex's "core", the center of the
    //vortex where no air is moving to prevent divide by zero issue.
    self.core = 0.05 * bird.Xl;
}


/*
  Returns the velocity of the air in a vortex at position (x,y,z)
  in the earth's frame. This value can be used to calculate drag forces on
  any birds this vortex interacts with.
*/
Vector3d Vortex::earth_vel(double x, double y, double z){
    Vortex & self = *this;
    double r = sqrt(sqr(y - self.pos[1]) + sqr(z - self.pos[2]));
    if (r < self.core || r > self.max_r)
	return Vector3d(0.0, 0.0, 0.0);

    double v_tan = self.gamma * sqr(r) / (2 * PI * r * (sqr(r) + sqr(self.core)));
    Vector3d r_vec = Vector3d(0.0, y - self.y(), z - self.z())/r;
    Vector3d tan_vec = v_tan * Vector3d(0.0, -r_vec[2], r_vec[1]);

    double phi = self.phi;
    double theta = self.theta;
    double psi = self.psi;
    Matrix3d mat = self.get_transform(phi, theta, psi);

    Vector3d a = matmul(mat, tan_vec);
    return -self.sign * a;
}

// velocity in the bird's frame
std::pair<Vector3d,Vector3d> Vortex::bird_vel(Bird & bird){
    Vortex & self = *this;
    Matrix3d mat = self.get_transform(bird.phi, bird.theta, bird.psi);
    Vector3d add = Vector3d(0.0, bird.Xl/2.0, 0.0);

    Vector3d pos_right = bird.xyz() +  matmul(mat, add);
    Vector3d pos_left = bird.xyz() -  matmul(mat, add);

    double r_right = sqrt(sqr(pos_right[1] - self.pos[1]) + sqr(pos_right[2] - self.pos[2]));
    double r_left = sqrt(sqr(pos_left[1] - self.pos[1]) + sqr(pos_left[2] - self.pos[2]));

    double v_tan_l = self.gamma * sqr(r_left) / (2 * PI * r_left * (sqr(r_left) + sqr(self.core)));
    double v_tan_r = self.gamma * sqr(r_right) / (2 * PI * r_right * (sqr(r_right) + sqr(self.core)));

    Vector3d r_vec_right = Vector3d(0.0, pos_right[1] - self.y(), pos_right[2] - self.z())/r_right;
    Vector3d r_vec_left = Vector3d(0.0, pos_left[1] - self.y(), pos_left[2] - self.z())/r_left;

    Vector3d tan_vec_left = v_tan_l * Vector3d(0.0, -r_vec_left[2], r_vec_left[1]);
    Vector3d tan_vec_right = v_tan_r * Vector3d(0.0, -r_vec_right[2], r_vec_right[1]);

    double phi = bird.phi - self.phi;
    double theta = bird.theta - self.theta;
    double psi = bird.psi - self.psi;
    mat = self.get_transform(phi, theta, psi);

    Vector3d left = matmul(mat, tan_vec_left);
    Vector3d right = matmul(mat, tan_vec_right);

    if (r_left < self.core || r_left > self.max_r)
	left = Vector3d(0.0, 0.0, 0.0);
    if (r_right < self.core || r_right > self.max_r)
	right = Vector3d(0.0, 0.0, 0.0);

    auto ret = std::make_pair(-self.sign * left, -self.sign * right);
    return ret;
}

Matrix3d Vortex::get_transform(double phi, double theta, double psi){
    double sphi = sin(phi);
    double cphi = cos(phi);
    double stheta = sin(theta);
    double ctheta = cos(theta);
    double spsi = sin(psi);
    double cpsi = cos(psi);

    Matrix3d mat = matrix(cphi * ctheta, -spsi * cphi + cpsi * stheta * sphi, spsi * sphi + cpsi * cphi * stheta,
			  spsi * ctheta, cpsi * cphi + sphi * stheta * spsi, -cpsi * sphi + stheta * spsi * cphi,
			  -stheta, ctheta * sphi, ctheta * cphi);
    return mat;
}
