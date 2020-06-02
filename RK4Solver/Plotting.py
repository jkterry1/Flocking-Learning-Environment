import numpy as np
import matplotlib.pyplot as plt

def plot(T, Y, agents):
    legend = []
    for i in agents:
        plt.plot(T, [row[0] for row in Y[i]])
        plt.plot(T, [row[1] for row in Y[i]])
        plt.plot(T, [row[2] for row in Y[i]])
        x = "{}: x".format(i)
        y = "{}: y".format(i)
        z = "{}: z".format(i)
        legend += [x,y,z]
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Bird Positions')
    plt.legend(legend)
    plt.show()
