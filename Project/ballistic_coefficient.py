import sim
import matplotlib.pyplot as plt
import numpy as np

file_path = 'Ballistic-Coefficient-Charts/'

time_step = 1
time_max = 5000
entry_altitude = 125000
entry_flight_path_angle = -15
entry_velocity = 6000

area = 1
ballistic_coefficients = [0.001,0.0037,0.01,0.037,0.1,0.37,1,3.7,10,37,100,207,400,1000,3700,10000,37000,100000,370000,1000000]  # kg/m^2
masses = [ballistic_coefficient * area for ballistic_coefficient in ballistic_coefficients]
terminal_velocities = []

annotation_indexes = [11,12]
annoation_labels = ['Curiosity','Starship']

aoa = 0

for mass in masses:
    sim_data = sim.simulate(
        time_step=time_step,
        time_max=time_max,
        mass=mass,
        area=area,
        aoa=aoa,
        entry_altitude=entry_altitude,
        entry_flight_path_angle=entry_flight_path_angle,
        entry_velocity=entry_velocity,
        verbose=False,
    )
    sim.plot(sim_data, filename=f"{file_path}Ballistic_Coefficient_BC-{str(mass).replace(".", "")}_aoa-{aoa}")
    terminal_velocities.append(sim_data['velocities'][-1])
    print(f"Mass: {mass} kg, Terminal Velocity: {sim_data['velocities'][-1]:.2f} m/s")

print("Ballistic Coefficient,Terminal Velocity")
for i, mass in enumerate(masses):
    print(f"{ballistic_coefficients[i]},{terminal_velocities[i]:.2f}")

# Plot in plt using log scale
plt.figure(figsize=(10, 6))
plt.plot(ballistic_coefficients, terminal_velocities, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Ballistic Coefficient (kg/m^2)')
plt.ylabel('Terminal Velocity (m/s)')
plt.title('Mars Entry Terminal Velocity vs Ballistic Coefficient')
plt.grid(True)

# add text to denote aoa eg."AoA: 20 [degree symbol]"
plt.text(0.70, 0.15, f"AoA: {aoa}°", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
plt.text(0.70, 0.1, f"Christopher Kalitin 2025", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')

for i, label in enumerate(annoation_labels):
    plt.annotate(label, xy=(ballistic_coefficients[annotation_indexes[i]], terminal_velocities[annotation_indexes[i]]), xytext=(ballistic_coefficients[annotation_indexes[i]], terminal_velocities[annotation_indexes[i]]*0.75))
plt.savefig(f'{file_path}ballistic_coefficient_vs_terminal_velocity_log_aoa{aoa}.png')
plt.close()

# now without log scale
plt.figure(figsize=(10, 6))
plt.plot(ballistic_coefficients, terminal_velocities, marker='o')
plt.xlabel('Ballistic Coefficient (kg/m^2)')
plt.ylabel('Terminal Velocity (m/s)')
plt.title('Mars Entry Terminal Velocity vs Ballistic Coefficient')
plt.grid(True)

# add text to denote aoa eg."AoA: 20 [degree symbol]"
plt.text(0.70, 0.15, f"AoA: {aoa}°", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
plt.text(0.70, 0.1, f"Christopher Kalitin 2025", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')

for i, label in enumerate(annoation_labels):
    plt.annotate(label, xy=(ballistic_coefficients[annotation_indexes[i]], terminal_velocities[annotation_indexes[i]]), xytext=(ballistic_coefficients[annotation_indexes[i]], terminal_velocities[annotation_indexes[i]]*0.75))
plt.savefig(f'{file_path}ballistic_coefficient_vs_terminal_velocity_linear_aoa{aoa}.png')
plt.close()
