import sim
import matplotlib.pyplot as plt
import numpy as np

time_step = 0.1
time_max = 5000
entry_altitude = 125000
entry_flight_path_angle = -15
entry_velocity = 6000

area = 10
masses = [0.001,0.01,0.1,1,10,100,1000,4444,10000,100000,1000000]  # kg
ballistic_coefficients = [mass / area for mass in masses]
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
        print_debug=False,
    )
    sim.plot(sim_data, filename=f"Ballistic-Coefficient-{str(mass).replace(".", "")}kgm2")
    terminal_velocities.append(sim_data['final_values']['velocity'])
    print(f"Mass: {mass} kg, Terminal Velocity: {sim_data['final_values']['velocity']:.2f} m/s")

# Plot in plt using log scale
plt.figure(figsize=(10, 6))
plt.plot(ballistic_coefficients, terminal_velocities, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Ballistic Coefficient (kg/m^2)')
plt.ylabel('Terminal Velocity (m/s)')
plt.title('Ballistic Coefficient vs Terminal Velocity')
plt.grid(True)
plt.annotate('Starship', xy=(masses[7], terminal_velocities[7]), xytext=(masses[7]*1.5, terminal_velocities[7]*1.5))
plt.savefig('ballistic_coefficient_vs_terminal_velocity_log.png')
plt.show()
plt.close()

# now without log scale
plt.figure(figsize=(10, 6))
plt.plot(ballistic_coefficients, terminal_velocities, marker='o')
plt.xlabel('Ballistic Coefficient (kg/m^2)')
plt.ylabel('Terminal Velocity (m/s)')
plt.title('Ballistic Coefficient vs Terminal Velocity')
plt.grid(True)
plt.annotate('Starship', xy=(masses[7], terminal_velocities[7]), xytext=(masses[7]*1.5, terminal_velocities[7]*1.5))
plt.savefig('ballistic_coefficient_vs_terminal_velocity_linear.png')
plt.show()
plt.close()
