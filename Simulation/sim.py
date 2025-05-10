import utils
import math
import matplotlib.pyplot as plt

def simulate(time_step=None, time_max=None, mass=None, area=None, entry_altitude=None, entry_flight_path_angle=None, entry_velocity=None, verbose=False):
    if time_step is None:
        time_step = 1 # seconds
    if time_max is None:
        time_max = 5000 # seconds
    if mass is None:
        mass = 1000 # kg
    if area is None:
        area = 50 # m^2
    if entry_altitude is None:
        entry_altitude = 125000
    if entry_flight_path_angle is None:
        entry_flight_path_angle = -15 # degrees
    if entry_velocity is None:
        entry_velocity = 6000 # m/s

    ballistic_coefficient = mass / area

    v_x = entry_velocity * math.cos(math.radians(entry_flight_path_angle))
    v_y = entry_velocity * math.sin(math.radians(entry_flight_path_angle))

    altitude = entry_altitude
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

    if verbose:
        print(f"Mass: {mass} kg")
        print(f"Area: {area} m^2")
        print(f"Ballistic Coefficient: {ballistic_coefficient} kg/m^2")
        print(f"Entry Altitude: {entry_altitude} m")
        print(f"Entry Flight Path Angle: {entry_flight_path_angle} degrees")
        print(f"Entry Velocity: {entry_velocity} m/s")
        print(f"Time Step: {time_step} seconds")
        print(f"Max Time: {time_max} seconds")

    t = 0
    while t < time_max and altitude > 0:
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
        
        flight_path_angle = math.degrees(math.atan2(v_y, v_x))
        
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
        
        if (verbose): print(f"t: {t:.2f}, altitude: {altitude:.2f}, downrange_distance: {downrange_distance:.2f}, velocity: {velocity:.2f}, v_x: {v_x:.2f}, v_y: {v_y:.2f}, a_x: {a_x:.2f}, a_y: {a_y:.2f}, net_acc: {net_acc:.2f}, drag_acc: {drag_acc:.2f}, grav_acc: {grav_acc:.2f}, flight_path_angle: {flight_path_angle:.2f}")
        
        t += time_step
    
    # Return all data in a dictionary
    return {
        'times': times,
        'altitudes': altitudes,
        'velocities': velocities,
        'flight_path_angles': flight_path_angles,
        'v_xs': v_xs,
        'v_ys': v_ys,
        'a_xs': a_xs,
        'a_ys': a_ys,
        'downrange_dists': downrange_dists,
        'net_accs': net_accs,
        'drag_accs': drag_accs,
        'grav_accs': grav_accs,
        'drag_coeffs': drag_coeffs,
        'atm_densities': atm_densities,
        'parameters': {
            'mass': mass,
            'area': area,
            'ballistic_coefficient': ballistic_coefficient,
            'entry_altitude': entry_altitude,
            'entry_flight_path_angle': entry_flight_path_angle,
            'entry_velocity': entry_velocity,
            'time_step': time_step,
            'time_max': time_max
        }
    }

def plot(data, filename='mars_entry_simulation.png', show=False, comparisons=None):
    # Comparisions is a list of tuples (velocity, altitude, label), MAKE SURE ITS A LIST, NOT JUST A TUPLE, USE THE SQUARE BRACKETS
    
    times = data['times']
    altitudes = data['altitudes']
    velocities = data['velocities']
    flight_path_angles = data['flight_path_angles']
    v_xs = data['v_xs']
    v_ys = data['v_ys']
    a_xs = data['a_xs']
    a_ys = data['a_ys']
    downrange_dists = data['downrange_dists']
    net_accs = data['net_accs']
    drag_accs = data['drag_accs']
    grav_accs = data['grav_accs']
    drag_coeffs = data['drag_coeffs']
    atm_densities = data['atm_densities']
    params = data['parameters']
    
    # Create figure
    plt.figure(figsize=(19.20, 10.80), dpi=100)
    plt.suptitle('Mars Entry Simulation', fontsize=16)

    # Plot 1,1: Altitude vs Time
    plt.subplot(3, 3, 1)
    plt.plot(times, altitudes)
    plt.title('Altitude vs Time')
    plt.ylabel('Altitude (m)')
    plt.grid(True)

    # Plot 1,2: Altitude vs. Velocity
    plt.subplot(3, 3, 2)
    plt.plot(velocities, altitudes, label='Simulation')
    if (comparisons is not None): 
        for comparison_vel_alt_label in comparisons:
            plt.plot(comparison_vel_alt_label[0], comparison_vel_alt_label[1], label=comparison_vel_alt_label[2], linestyle='--')
    plt.title('Altitude vs Velocity')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Altitude (m)')
    plt.legend()
    plt.grid(True)

    # Plot 1,3: Altitude vs Downrange Distance
    plt.subplot(3, 3, 3)
    plt.plot(downrange_dists, altitudes)
    plt.title('Altitude vs Downrange Distance')
    plt.xlabel('Downrange Distance (m)')
    plt.ylabel('Altitude (m)')
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

    # Plot 2,2: ax, ay, anet vs Time
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

    # Plot 3,1: Drag Coefficient and Atmospheric Density vs Time
    plt.subplot(3, 3, 7)
    ax5_1 = plt.gca()
    ax5_1.plot(times, drag_coeffs, 'b-', label='Drag Coefficient')
    ax5_1.set_title('Drag Coefficient and Atmospheric Density vs Time')
    ax5_1.set_ylabel('Drag Coefficient', color='b')
    ax5_1.tick_params(axis='y', labelcolor='b')
    ax5_1.grid(True)

    ax5_2 = ax5_1.twinx()
    ax5_2.plot(times, atm_densities, 'r-', label='Atmospheric Density')
    ax5_2.set_ylabel('Density (kg/m³)', color='r')
    ax5_2.tick_params(axis='y', labelcolor='r')

    lines1, labels1 = ax5_1.get_legend_handles_labels()
    lines2, labels2 = ax5_2.get_legend_handles_labels()
    ax5_1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Plot 3,2: Flight Path Angle vs Time
    plt.subplot(3, 3, 8)
    plt.plot(times, flight_path_angles)
    plt.title('Flight Path Angle vs Time')
    plt.ylabel('Angle (degrees)')
    plt.grid(True)

    # Plot 3,3: Parameters
    plt.subplot(3, 3, 9)
    plt.axis('off')
    plt.text(-0.15, 1.0, f"Mass: {params['mass']} kg", fontsize=10)
    plt.text(-0.15, 0.93, f"Area: {params['area']} m²", fontsize=10)
    plt.text(-0.15, 0.86, f"Ballistic Coefficient: {params['ballistic_coefficient']:.2f} kg/m²", fontsize=10)
    
    plt.text(-0.15, 0.76, f"Entry Altitude: {params['entry_altitude']} m", fontsize=10)
    plt.text(-0.15, 0.69, f"Entry Flight Path Angle: {params['entry_flight_path_angle']} degrees", fontsize=10)
    plt.text(-0.15, 0.62, f"Entry Velocity: {params['entry_velocity']} m/s", fontsize=10)
    
    plt.text(-0.15, 0.52, f"Time Step: {params['time_step']} second(s)", fontsize=10)
    plt.text(-0.15, 0.45, f"Max Time: {params['time_max']} seconds", fontsize=10)
    
    plt.text(-0.15, 0.14, f"Christopher Kalitin 2025", fontsize=10)

    # Final values
    plt.text(0.4, 1.0, f"Final Altitude: {altitudes[-1]:.2f} m", fontsize=10)
    plt.text(0.4, 0.93, f"Final Downrange Distance: {downrange_dists[-1]:.2f} m", fontsize=10)
    
    # refactor all below to use -1 index instead of final
    plt.text(0.4, 0.86, f"Final Velocity: {velocities[-1]:.2f} m/s", fontsize=10)
    plt.text(0.4, 0.79, f"Final Horizontal Velocity: {v_xs[-1]:.2f} m/s", fontsize=10)
    plt.text(0.4, 0.72, f"Final Vertical Velocity: {v_ys[-1]:.2f} m/s", fontsize=10)
    
    plt.text(0.4, 0.62, f"Final Acceleration: {net_accs[-1]:.2f} m/s²", fontsize=10)
    plt.text(0.4, 0.55, f"Final Horizontal Acceleration: {a_xs[-1]:.2f} m/s²", fontsize=10)
    plt.text(0.4, 0.48, f"Final Vertical Acceleration: {a_ys[-1]:.2f} m/s²", fontsize=10)
    plt.text(0.4, 0.41, f"Final Drag Acceleration: {drag_accs[-1]:.2f} m/s²", fontsize=10)
    plt.text(0.4, 0.34, f"Final Gravity Acceleration: {grav_accs[-1]:.2f} m/s²", fontsize=10)
    
    plt.text(0.4, 0.24, f"Final Flight Path Angle: {flight_path_angles[-1]:.2f} degrees", fontsize=10)
    plt.text(0.4, 0.17, f"Final Time: {times[-1]:.2f} seconds", fontsize=10)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(left=0.055, right=0.98, top=0.925, bottom=0.032, hspace=0.29, wspace=0.31)

    # Save the plot
    plt.savefig(filename)
    if show: plt.show()
    plt.close()

# Example usage
if __name__ == "__main__":
    sim_data = simulate(
        time_step=0.1,
        time_max=5000,
        mass=3300,
        area=15.9,
        entry_altitude=125000,
        entry_flight_path_angle=-14,
        entry_velocity=5800
    )
    plot(sim_data)