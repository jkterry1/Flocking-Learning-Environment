from energy_meter import energy
import matplotlib.pyplot as plt

# energies = []
# for i in range(1, 101):
#     v = [float(i), 0.0, 0.0]
#     energies.append(energy(v))
# plt.plot(energies)
# plt.xlabel("Forward Velocity (m/s)")
# plt.ylabel("Energy (J)")
# plt.title("Estimated Energy to Travel 1m")
# plt.show()

v = [20.0, 0.0, 0.0]
print("energy: ", energy(v))
