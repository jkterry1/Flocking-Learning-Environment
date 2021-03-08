## Run

install: `pip install pettingzoo pybind11`

compile: `bash build.sh`

run example: `python solver_tester`

NOTE: need to compile before running python code

### Code outline

Python:

* `flocking_env.py`: The PettingZoo environment.
* `plotting.py`: Plot utilities
* `solver_tester.py`: Tests bird flocking
* `test_flocking_api.py`: Tests the PettingZoo environment API (unmaintained as of Mar 7)

C++:

* `basic.cpp`: C++ testing code
* `bird.cpp`: Bird data structure, bird history
* `DiffEqs.cpp`: RK4 Differential equation solver
* `flock.cpp`: Environment level data structure
* `py_interface.cpp`: Python interface for environment
* `vortex.cpp`: Vortex data structure

### Environment Parameters
- **N**: number of birds (if bird_inits is None)
- **h**: seconds per frame (step size)
- **t**: maximum seconds per episode
- **energy_reward**: the reward for a bird using energy (negative to -incentivize limiting energy use)
- **forward_reward**: the reward for a bird moving forward
- **crash_reward**: the reward for a bird crashing into another bird or the ground
- **max_observable_birds**: the number of neighboring birds a bird can see
- **bird_inits**: initial positions of the birds (None for default random sphere)
- **LIA**: boolean choice to include Local approximation for vortice movement
- **bird_filename**: the file you want to log the bird states in
- **vortex_filename**: the file you want to log the vortex states in
- **vortex_update_frequency**: Period of adding new points on the vortex line.

### Actions
The action space is a 5-vector where each index represents:
   - 0: Thrust, a forward push on the bird
   - 1: alpha rotation of left wing (in degrees)
   - 2: beta rotation of left wing (in degrees)
   - 3: alpha rotation of right wing (in degrees)
   - 4: beta rotation of right wing (in degrees)

### Observations
Observation space is a vector with
    22 dimensions for the current bird's state and
    6 dimensions for each of the birds the current bird can observe.

   Bird's state observations:
  - 0-2:    Force (N) on the bird in each direction (fu, fv, fw)
  - 3-5:    Torque (N m) on the bird in each direction (Tu, Tv, Tw)
   - 6-8:    Bird's position (m) in each dimension (x, y, z)
   - 9-11:   Bird's velocity (m/s) in each direction (u, v, w)
   - 12-14:  Bird's angular velocity (degrees/s) in each direction (p, q, r)
   - 15-17:  Bird's orientation (degrees) (phi, theta, psi)
   - 18-19:  Left wing orientation (degrees) (alpha, beta)
   - 20-21:  Right wing orientation (degrees) (alpha, beta)

 Following this, starting at observation 22, there will be
 6-dimension vectors for each bird the current bird can observe.
Each of these vectors contains:
- 0-2:    Relative position to the current bird
- 3-5:    Relative velocity to the current bird

### Bird Values
A bird's position, orientation, and movement are defined by several variables.

- **Position**: A bird's position vector is defined by **(x, y, z)**. This is the bird's x, y, and z position in the Earth's frame. The position of the center of the bird is always (0,0,0) in the bird's frame.

- **Orientation**: A bird's orientation is defined by 3 Euler angles, **(phi, theta, psi)**.  Each of these angles is a rotation (in degrees) around an axis in the bird frame.
- **Velocity**: A bird's velocity is defined by **(u, v, w)**. These are the values for velocity along the 3 main axes in the bird frame. 
![image](Images/pos-ori-vel.jpeg)

- **Rotational Velocity**: A bird's angular velocity is defined by **(p, q, r)**. These are the rates of rotation around each of the 3 main axes in the bird's frame. 

- **Wing Orientation**: The wings can be adjusted in two ways, rotated forward and backward (**alpha**) or rotated up and down (**beta**).
![image](Images/wings.jpeg)


### Simulation Equations

The position, velocity, Euler angles, and angular velocity are calculated every time step using a numerical differential equations solver. Each of these properties is described by a differential equation in time. Here are the equations that describe each type of motion. 

**Position**

The "E" subscript indicates that this value is in the earth's frame. Unless otherwise stated, all other values are in the bird's frame
The equations are shortened, s is sine,c is cosine, and t is tangent.
![image](Images/diffeq-pos.jpeg)

**Velocity**

m is the bird's mass. 
(X, Y, Z) is the force the bird experiences in each direction.
![image](Images/diffeq-vel.jpeg)

**Angles**

![image](Images/diffeq-angles.jpeg)

**Angular Velocity**

(L, M, N) is the torque in the (u,v,w) direction.
Ix, Iy, Iz are the moments of inertia
![image](Images/diffeq-angvel.jpeg)

### Vortices

A vortex is defined by a line in space that represents the center of the vortex at each point along the line. Birds will "drop" new points on this line after a set number of frames, and points will go out of existence after a certain amount of time as the vortices lose strength and disappear.

![image](Images/vortex-line.jpeg)

When a bird interacts with a vortex, it searches for the nearest defined point on the vortex line and uses the strength and orientation from that point to calculate any forces due to the vortex.

The velocity of air caused by the vortex is defined by the Burnham-Hallock model. This is a velocity in the direction of air motion (tangent velocity).

Parameters:
- Gamma: the strength of the vortex
- r: the distance from the center of the vortex
- rc: the size of the "vortex core", an area with no motion at the center of the vortex.

![image](Images/vortex-vel.jpeg)



**Local Induction Approximation**

The local induction approximation (LIA) is a differential equation that describes how the vortex line induces motion in itself due to its own curvature.

Here are the equations that describe this motion:
![image](Images/LIA-theory.jpeg)

This is for the continuous case, but the vortices in the simulation are defined by discrete points, so a discretized version of these equations is used instead.

The simulation calculates new values for points from oldest to newest, using the last calculated point to determine the value of the next point. 

This diagram shows the values needed from each point.
- n is the last point calculated on the line
- n+1 is the next point on the line
- r: position in space
- t: the tangent vector
- n: the normal vector
- b: the binormal vector
![image](Images/lia-values.jpeg)

Using these calculated values, the final velocity of a point on the vortex line is:
![image](Images/lia-discrete.jpeg)
Where epsilon is the distance between points.

This velocity is then used to calculate position using the same numerical differential equation solver as used in the rest of the project. 