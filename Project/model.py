import utils
import math
import matplotlib.pyplot as plt

time_step = 1 # seconds
time_max = 200 # seconds

mass = 1000 # kg
area = 10 # m^2
ballistic_coefficient = mass/area # kg/m^2

altitude = 125000
entry_flight_path_angle = -15 # degrees
entry_velocity = 6000 # m/s

v_x = entry_velocity * math.cos(math.radians(entry_flight_path_angle))
v_y = entry_velocity * math.sin(math.radians(entry_flight_path_angle))

velocity = entry_velocity
flight_path_angle = entry_flight_path_angle

downrange_distance = 0

times = []
altitudes = []
velocities = []
flight_path_angles = []
v_xs = []
v_ys = []
a_xs = []
a_ys = []
downrange_dists = []
net_accs = []
drag_accs = []
grav_accs = []
drag_coeffs = []
atm_densities = []

for t in range(0, time_max, time_step):
    atm_density = utils.get_atmospheric_density(altitude)
    drag_coeff = utils.get_interpolated_drag_coefficient(velocity)
    drag_acc = utils.get_drag_acc(mass, velocity, area, drag_coeff, atm_density)
    
    grav_acc = utils.get_gravity_acc(altitude)
    
    a_x = drag_acc * math.cos(math.radians(flight_path_angle))
    a_y = drag_acc * math.sin(math.radians(flight_path_angle)) + grav_acc
    
    v_x += a_x * time_step
    v_y += a_y * time_step
    
    net_acc = math.sqrt(a_x**2 + a_y**2)
    velocity = math.sqrt(v_x**2 + v_y**2)
    
    downrange_distance += v_x * time_step
    downrange_dists.append(downrange_distance)
    
    altitude += v_y * time_step
    
    flight_path_angle = math.degrees(math.atan(v_y / v_x))
    
    times.append(t)
    altitudes.append(altitude)
    velocities.append(velocity)
    flight_path_angles.append(flight_path_angle)
    v_xs.append(v_x)
    v_ys.append(v_y)
    a_xs.append(a_x)
    a_ys.append(a_y)
    net_accs.append(net_acc)
    drag_accs.append(drag_acc)
    grav_accs.append(grav_acc)
    drag_coeffs.append(drag_coeff)
    atm_densities.append(atm_density)
    
    # make all values equally spaced
    print(f"t: {t}, alt: {altitude:.2f} m, vel: {velocity:.2f} m/s, vx: {v_x:.2f} m/s, vy: {v_y:.2f} m/s, flight path angle: {flight_path_angle:.2f} degrees, drag acc: {drag_acc:.2f} m/s^2, grav acc: {grav_acc:.2f} m/s^2, drag coeff: {drag_coeff:.2f}")
    
    if altitude < 0:
        print("Landed!")
        break

# Plot everything
# all of these (combining graphs where appropriate) should be in one figure:
"""
times.append(t)
altitudes.append(altitude)
velocities.append(velocity)
flight_path_angles.append(flight_path_angle)
v_xs.append(v_x)
v_ys.append(v_y)
drag_accs.append(drag_acc)
grav_accs.append(grav_acc)
drag_coeffs.append(drag_coeff)
atm_densities.append(atm_density)
"""

plt.figure(figsize=(12, 12))

plt.suptitle('Mars Entry Simulation', fontsize=16)

# Plot 1,1: Altitude vs Time
plt.subplot(3, 3, 1)
plt.plot(times, altitudes)
plt.title('Altitude vs Time')
plt.ylabel('Altitude (m)')
plt.grid(True)

# Plot 1,2: Altitude vs Downrange Distance
plt.subplot(3, 3, 2)
plt.plot(downrange_dists, altitudes)
plt.title('Altitude vs Downrange Distance')
plt.ylabel('Altitude (m)')
plt.grid(True)

# Plot 1,3: Downrange Distance vs Time
plt.subplot(3, 3, 3)
plt.plot(times, downrange_dists)
plt.title('Downrange Distance vs Time')
plt.ylabel('Downrange Distance (m)')
plt.grid(True)

# Plot 2,1: Velocities vs Time
plt.subplot(3, 3, 4)
plt.plot(times, velocities, label='Total Velocity')
plt.plot(times, v_xs, label='Horizontal Velocity')
plt.plot(times, v_ys, label='Vertical Velocity')
plt.title('Velocities vs Time')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)

# Plot 2,2: ax,ay,anet vs Time
plt.subplot(3, 3, 5)
plt.plot(times, a_xs, label='Horizontal Acceleration')
plt.plot(times, a_ys, label='Vertical Acceleration')
plt.plot(times, net_accs, label='Net Acceleration')
plt.title('Accelerations vs Time')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.grid(True)

# Plot 2,3: drag_acc, grav_acc, net acc vs Time
plt.subplot(3, 3, 6)
plt.plot(times, drag_accs, label='Drag Acceleration')
plt.plot(times, grav_accs, label='Gravity Acceleration')
plt.plot(times, net_accs, label='Net Acceleration')
plt.title('Drag and Gravity Acceleration vs Time')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.grid(True)

# Plot 3,1:Drag Coefficient and Atmospheric Density vs Time
plt.subplot(3, 3, 7)
ax5_1 = plt.gca()
ax5_1.plot(times, drag_coeffs, 'b-', label='Drag Coefficient')
ax5_1.set_title('Drag Coefficient and Atmospheric Density vs Time')
ax5_1.set_ylabel('Drag Coefficient', color='b')
ax5_1.tick_params(axis='y', labelcolor='b')
ax5_1.grid(True)

ax5_2 = ax5_1.twinx()  # Create a twin y-axis
ax5_2.plot(times, atm_densities, 'r-', label='Atmospheric Density')
ax5_2.set_ylabel('Density (kg/m³)', color='r')
ax5_2.tick_params(axis='y', labelcolor='r')

# Add a legend for both axes
lines1, labels1 = ax5_1.get_legend_handles_labels()
lines2, labels2 = ax5_2.get_legend_handles_labels()
ax5_1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# Plot 3,2: Flight Path Angle vs Time
plt.subplot(3, 3, 8)
plt.plot(times, flight_path_angles)
plt.title('Flight Path Angle vs Time')
plt.ylabel('Angle (degrees)')
plt.grid(True)

plt.tight_layout()
plt.subplots_adjust(left=0.055, right=0.98, top=0.925, bottom=0.032, hspace=0.24, wspace=0.31)

plt.show()