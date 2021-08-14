import matplotlib.pyplot as plt

phi = []
theta = []
psi = []
alpha_l = []
beta_l = []
alpha_r = []
beta_r = []

file = open('log_birds_6.csv', 'r')
file = open('bird_log_1.csv', 'r')
for line in file:
    if line[0] == '0':
        line = line.split(",")
        phi.append(line[5])
        theta.append(line[6])
        psi.append(line[7])
        alpha_l.append(line[8])
        beta_l.append(line[10])
        alpha_r.append(line[9])
        beta_r.append(line[11])


plt.plot(phi)
plt.plot(theta)
plt.plot(psi)
plt.legend(['phi', 'theta', 'psi'])
plt.show()

plt.plot(alpha_l)
plt.plot(alpha_r)
plt.plot(beta_l)
plt.plot(beta_r)
plt.legend(['alpha_l', 'alpha_r', 'beta_l', 'beta_r'])
plt.show()
