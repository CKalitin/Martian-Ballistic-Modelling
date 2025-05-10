import sim
import matplotlib.pyplot as plt
import numpy as np

file_path = 'Ballistic-Coefficient-Charts/'

time_step = 0.1
time_max = 5000
entry_altitude = 125000
entry_flight_path_angle = -15
entry_velocity = 6000

area = 1
ballistic_coefficients = [0.001,0.01,0.1,1,10,100,207,400,1000,10000,100000,1000000]  # kg/m^2
masses = [ballistic_coefficient * area for ballistic_coefficient in ballistic_coefficients]
terminal_velocities = []

for mass in masses:
    sim_data = sim.simulate(
        time_step=time_step,
        time_max=time_max,
        mass=mass,
        area=area,
        entry_altitude=entry_altitude,
        entry_flight_path_angle=entry_flight_path_angle,
        entry_velocity=entry_velocity,
        verbose=False,
    )
    sim.plot(sim_data, filename=f"{file_path}Ballistic-Coefficient-{str(mass).replace(".", "")}kgm2")
    terminal_velocities.append(sim_data['final_values']['velocity'])
    print(f"Mass: {mass} kg, Terminal Velocity: {sim_data['final_values']['velocity']:.2f} m/s")

print("Ballistic Coefficient,Terminal Velocity")
for i, mass in enumerate(masses):
    print(f"{ballistic_coefficients[i]},{terminal_velocities[i]:.2f}")

annotation_indexes = [5,7]
annoation_labels = ['Curiosity','Starship']

# Plot in plt using log scale
plt.figure(figsize=(10, 6))
plt.plot(ballistic_coefficients, terminal_velocities, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Ballistic Coefficient (kg/m^2)')
plt.ylabel('Terminal Velocity (m/s)')
plt.title('Mars Entry Terminal Velocity vs Ballistic Coefficient')
plt.grid(True)
for i, label in enumerate(annoation_labels):
    plt.annotate(label, xy=(ballistic_coefficients[annotation_indexes[i]], terminal_velocities[annotation_indexes[i]]), xytext=(ballistic_coefficients[annotation_indexes[i]], terminal_velocities[annotation_indexes[i]]*0.75))
plt.savefig(f'{file_path}ballistic_coefficient_vs_terminal_velocity_log.png')
plt.close()

# now without log scale
plt.figure(figsize=(10, 6))
plt.plot(ballistic_coefficients, terminal_velocities, marker='o')
plt.xlabel('Ballistic Coefficient (kg/m^2)')
plt.ylabel('Terminal Velocity (m/s)')
plt.title('Mars Entry Terminal Velocity vs Ballistic Coefficient')
plt.grid(True)
for i, label in enumerate(annoation_labels):
    plt.annotate(label, xy=(ballistic_coefficients[annotation_indexes[i]], terminal_velocities[annotation_indexes[i]]), xytext=(ballistic_coefficients[annotation_indexes[i]], terminal_velocities[annotation_indexes[i]]*0.75))
plt.savefig(f'{file_path}ballistic_coefficient_vs_terminal_velocity_linear.png')
plt.close()
