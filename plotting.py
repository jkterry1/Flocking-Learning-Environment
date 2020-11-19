import matplotlib.pyplot as plt
import numpy as np

def close():
    plt.close()

def plot_values(birds, show = True):
    fig = plt.figure()
    plt1 = fig.add_subplot(411)
    plt2 = fig.add_subplot(412)
    plt3 = fig.add_subplot(413)
    plt4 = fig.add_subplot(414)
    for bird in range(len(birds)):
        bird = birds[bird]
        t = np.arange(len(bird.U))

        plt1.title.set_text('position')
        plt1.set_xlabel('Time (s)')
        plt1.set_ylabel('m')
        plt1.plot(t, bird.X)
        plt1.plot(t, bird.Y)
        plt1.plot(t, bird.Z)
        leg = []
        for _ in birds:
            leg += ['x', 'y', 'z']
        plt1.legend(leg)

        plt2.title.set_text('angular vel')
        plt2.set_xlabel('Time (s)')
        plt2.set_ylabel('rad/s')
        plt2.plot(t, bird.P)
        plt2.plot(t, bird.Q)
        plt2.plot(t, bird.R)
        leg = []
        for _ in birds:
            leg += ['p', 'q', 'r']
        plt2.legend(leg)

        plt3.title.set_text('angle')
        plt3.set_xlabel('Time (s)')
        plt3.set_ylabel('rad')
        plt3.plot(t, bird.PHI)
        plt3.plot(t, bird.THETA)
        plt3.plot(t, bird.PSI)
        leg = []
        for _ in birds:
            leg += ['phi', 'theta', 'psi']
        plt3.legend(leg)

        plt4.title.set_text('velocity')
        plt4.set_xlabel('Time (s)')
        plt4.set_ylabel('(m/s)')
        plt4.plot(t, bird.U)
        plt4.plot(t, bird.V)
        plt4.plot(t, bird.W)
        leg = []
        for _ in birds:
            leg += ['u', 'v', 'w']
        plt4.legend(leg)

    if show:
        plt.show()

def plot_birds(birds, plot_vortices = False, show = True):
    first = plot_vortices
    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    for b in range(len(birds)):
        bird = birds[b]
        ax.plot(xs = bird.X, ys = bird.Y, zs = bird.Z, zdir = 'z', color = 'orange')
        ax.scatter([v.pos[0] for v in bird.VORTICES_LEFT],
                    [v.pos[1] for v in bird.VORTICES_LEFT],
                    [v.pos[2] for v in bird.VORTICES_LEFT],
                    color = 'red', s = .5)
        ax.scatter([v.pos[0] for v in bird.VORTICES_RIGHT],
                    [v.pos[1] for v in bird.VORTICES_RIGHT],
                    [v.pos[2] for v in bird.VORTICES_RIGHT],
                    color = 'red', s = .5)
        ax.scatter([bird.X[0]], [bird.Y[0]], [bird.Z[0]], 'blue')

    x = []; y = []; z = []
    u = []; v = []; w = []
    r = 0.25
    if first:
        first = False
        for vort in bird.VORTICES_LEFT:
            x.append(vort.x);y.append(vort.y + r);z.append(vort.z + r)
            a,b,c = vort.earth_vel(vort.x, vort.y + r, vort.z + r)
            u.append(a); v.append(b); w.append(c)

            x.append(vort.x);y.append(vort.y - r);z.append(vort.z - r)
            a,b,c = vort.earth_vel(vort.x, vort.y - r, vort.z - r)
            u.append(a);v.append(b);w.append(c)

            x.append(vort.x);y.append(vort.y);z.append(vort.z + r)
            a,b,c = vort.earth_vel(vort.x, vort.y, vort.z + r)
            u.append(a); v.append(b); w.append(c)

            x.append(vort.x);y.append(vort.y);z.append(vort.z - r)
            a,b,c = vort.earth_vel(vort.x, vort.y, vort.z - r)
            u.append(a); v.append(b); w.append(c)

            x.append(vort.x);y.append(vort.y - r);z.append(vort.z + r)
            a,b,c = vort.earth_vel(vort.x, vort.y - r, vort.z + r)
            u.append(a); v.append(b); w.append(c)

            x.append(vort.x);y.append(vort.y + r);z.append(vort.z - r)
            a,b,c = vort.earth_vel(vort.x, vort.y + r, vort.z - r)
            u.append(a);v.append(b);w.append(c)

        for vort in bird.VORTICES_RIGHT:
            x.append(vort.x);y.append(vort.y + r);z.append(vort.z + r)
            a,b,c = vort.earth_vel(vort.x, vort.y + r, vort.z + r)
            u.append(a); v.append(b); w.append(c)

            x.append(vort.x);y.append(vort.y - r);z.append(vort.z - r)
            a,b,c = vort.earth_vel(vort.x, vort.y - r, vort.z - r)
            u.append(a);v.append(b);w.append(c)

            x.append(vort.x);y.append(vort.y);z.append(vort.z + r)
            a,b,c = vort.earth_vel(vort.x, vort.y, vort.z + r)
            u.append(a); v.append(b); w.append(c)

            x.append(vort.x);y.append(vort.y);z.append(vort.z - r)
            a,b,c = vort.earth_vel(vort.x, vort.y, vort.z - r)
            u.append(a); v.append(b); w.append(c)

            x.append(vort.x);y.append(vort.y - r);z.append(vort.z + r)
            a,b,c = vort.earth_vel(vort.x, vort.y - r, vort.z + r)
            u.append(a); v.append(b); w.append(c)

            x.append(vort.x);y.append(vort.y + r);z.append(vort.z - r)
            a,b,c = vort.earth_vel(vort.x, vort.y + r, vort.z - r)
            u.append(a);v.append(b);w.append(c)



    ax.quiver(x,y,z,u,v,w, length = .1, normalize = True)
    if show:
        plt.show()
