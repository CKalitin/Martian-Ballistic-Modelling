import utils
import math
import matplotlib.pyplot as plt
import numpy as np
import time

def simulate(time_step=None, time_max=None, mass=None, area=None, aoa=None, entry_altitude=None, entry_flight_path_angle=None, entry_velocity=None, verbose=False):
    start_time = time.time()

    if type(aoa) != list:
        aoa = [[entry_altitude, aoa], [0, aoa]]
    aoa_list = utils.get_numpy_aoa_list(aoa)

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
    angles_of_attack = []
    v_xs = []
    v_ys = []
    a_xs = []
    a_ys = []
    downrange_dists = []
    net_accs = []
    drag_accs = []
    lift_accs = []
    grav_accs = []
    drag_coeffs = []
    atm_pressures = []
    atm_temperatures = []
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
        atm_pressure = utils.get_atmospheric_pressure(altitude) # Just for printing
        atm_temperature = utils.get_temperature(altitude) # Just for printing
        atm_density = utils.get_atmospheric_density_other(altitude, atm_pressure, atm_temperature) # Be sure not to use the formula that requires pressure and temp
        drag_coeff = utils.get_interpolated_drag_coefficient(velocity)
        drag_acc = utils.get_drag_acc(mass, velocity, area, drag_coeff, atm_density)
        
        aoa = utils.get_interpolated_aoa(aoa_list, altitude)
        angles_of_attack.append(aoa)
        
        lift_to_drag_ratio = utils.get_interpolated_lift_to_drag_ratio(aoa)
        lift_acc = utils.get_lift_acc(drag_acc, lift_to_drag_ratio)
        
        grav_acc = utils.get_gravity_acc(altitude)
        
        a_x = drag_acc * math.cos(math.radians(flight_path_angle)) + lift_acc * math.cos(math.radians(flight_path_angle+90))
        a_y = drag_acc * math.sin(math.radians(flight_path_angle)) + lift_acc * math.sin(math.radians(flight_path_angle+90)) + grav_acc
        
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
        lift_accs.append(lift_acc)
        grav_accs.append(grav_acc)
        drag_coeffs.append(drag_coeff)
        atm_pressures.append(atm_pressure)
        atm_temperatures.append(atm_temperature)
        atm_densities.append(atm_density)
        
        if (verbose): print(f"t: {t:.2f}, altitude: {altitude:.2f}, downrange_distance: {downrange_distance:.2f}, velocity: {velocity:.2f}, v_x: {v_x:.2f}, v_y: {v_y:.2f}, a_x: {a_x:.2f}, a_y: {a_y:.2f}, net_acc: {net_acc:.2f}, drag_acc: {drag_acc:.2f}, grav_acc: {grav_acc:.2f}, flight_path_angle: {flight_path_angle:.2f}")
        
        t += time_step
    
    execution_time = time.time() - start_time
    
    # Return all data in a dictionary
    return {
        'times': times,
        'altitudes': altitudes,
        'velocities': velocities,
        'flight_path_angles': flight_path_angles,
        'angles_of_attack': angles_of_attack,
        'v_xs': v_xs,
        'v_ys': v_ys,
        'a_xs': a_xs,
        'a_ys': a_ys,
        'downrange_dists': downrange_dists,
        'net_accs': net_accs,
        'drag_accs': drag_accs,
        'lift_accs': lift_accs,
        'grav_accs': grav_accs,
        'drag_coeffs': drag_coeffs,
        'atm_pressures': atm_pressures,
        'atm_temperatures': atm_temperatures,
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
        },
        'execution_time': execution_time
    }

def plot(data, title="Mars Entry Simulation", filename='mars_entry_simulation.png', show=False, comparisons=None):
    # Comparisions is a list of tuples (velocity, altitude, label), MAKE SURE ITS A LIST, NOT JUST A TUPLE, USE THE SQUARE BRACKETS
    
    times = data['times']
    altitudes = data['altitudes']
    velocities = data['velocities']
    flight_path_angles = data['flight_path_angles']
    angles_of_attack = data['angles_of_attack']
    v_xs = data['v_xs']
    v_ys = data['v_ys']
    a_xs = data['a_xs']
    a_ys = data['a_ys']
    downrange_dists = data['downrange_dists']
    net_accs = data['net_accs']
    drag_accs = data['drag_accs']
    lift_accs = data['lift_accs']
    grav_accs = data['grav_accs']
    drag_coeffs = data['drag_coeffs']
    atm_pressures = data['atm_pressures']
    atm_temperatures = data['atm_temperatures']
    atm_densities = data['atm_densities']
    params = data['parameters']
    execution_time = data['execution_time']
    
    # Create figure
    plt.figure(figsize=(19.20, 10.80), dpi=100)
    plt.suptitle(title, fontsize=16)
    plt.gcf().text(0.01, 0.965, f"Christopher Kalitin 2025", fontsize=12)
    # no axis
    plt.axis('off')

    # Plot 1,1: Altitude vs Time
    plt.subplot(3, 3, 1)
    plt.plot(times, altitudes, label='Simulation', zorder=999)
    for comparison in comparisons:
        if 'AltVsTime-time' in comparison:
            plt.plot(comparison['AltVsTime-time'], comparison['AltVsTime-alt'], '--', label=comparison['label'])
    plt.title('Altitude vs Time')
    plt.ylabel('Altitude (m)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    # Plot 1,2: Altitude vs. Velocity
    plt.subplot(3, 3, 2)
    plt.subplot(3, 3, 2)
    plt.plot(velocities, altitudes, label='Simulation', zorder=999)
    for comparison in comparisons:
        if 'AltVsVel-vel' in comparison:
            plt.plot(comparison['AltVsVel-vel'], comparison['AltVsVel-alt'], '--', label=comparison['label'])
    plt.title('Altitude vs Velocity')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Altitude (m)')
    plt.legend()
    plt.grid(True)

    # Plot 1,3: Altitude vs Downrange Distance
    plt.subplot(3, 3, 3)
    plt.plot(downrange_dists, altitudes, label='Simulation', zorder=999)
    for comparison in comparisons:
        if 'AltVsDownrangeDist-dist' in comparison:
            plt.plot(comparison['AltVsDownrangeDist-dist'], comparison['AltVsDownrangeDist-alt'], '--', label=comparison['label'])
    plt.title('Altitude vs Downrange Distance')
    plt.xlabel('Downrange Distance (m)')
    plt.ylabel('Altitude (m)')
    plt.grid(True)
    
    # Plot 2,1: Velocities vs Time
    plt.subplot(3, 3, 4)
    plt.plot(times, velocities, label='Total Velocity', zorder=999)
    plt.plot(times, v_xs, label='Horizontal Velocity', zorder=998)
    plt.plot(times, v_ys, label='Vertical Velocity', zorder=997)
    for comparison in comparisons:
        if 'VelVsTime-time' in comparison:
            plt.plot(comparison['VelVsTime-time'], comparison['VelVsTime-vel'], '--', label=comparison['VelVsTime-label'])
        if 'VVelVsTime-vel' in comparison:
            plt.plot(comparison['VVelVsTime-time'], comparison['VVelVsTime-vel'], '--', label=comparison['VVelVsTime-label'])
        if 'HVelVsTime-vel' in comparison:
            plt.plot(comparison['HVelVsTime-time'], comparison['HVelVsTime-vel'], '--', label=comparison['HVelVsTime-label'])
    plt.title('Velocities vs Time')
    plt.ylabel('Velocity (m/s)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    # Plot 2,2: ax, ay, anet vs Time
    plt.subplot(3, 3, 5)
    plt.plot(times, a_xs, label='Horizontal Acceleration')
    plt.plot(times, a_ys, label='Vertical Acceleration')
    plt.plot(times, net_accs, label='Net Acceleration')
    plt.title('Accelerations vs Time')
    plt.ylabel('Acceleration (m/s²)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    # Plot 2,3: drag_acc, lift_acc, grav_acc, net acc vs Time
    plt.subplot(3, 3, 6)
    plt.plot(times, drag_accs, label='Drag Acceleration')
    plt.plot(times, lift_accs, label='Lift Acceleration')
    plt.plot(times, grav_accs, label='Gravity Acceleration')
    plt.plot(times, net_accs, label='Net Acceleration')
    plt.title('Drag, Lift, Gravity Acceleration vs Time')
    plt.ylabel('Acceleration (m/s²)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    # Plot 3,1: Flight Path Angle and Angle of Attack vs Time
    plt.subplot(3, 3, 7)
    plt.plot(times, flight_path_angles, label='Flight Path Angle')
    plt.plot(times, angles_of_attack, label='Angle of Attack')
    if comparisons:
        for comparison in comparisons:
            if 'AoAVsTime-time' in comparison:
                if 'AoAVsTime-label' in comparison:
                    plt.plot(comparison['AoAVsTime-time'], comparison['AoAVsTime-aoa'], '--', label=comparison['AoAVsTime-label'])
                else:
                    plt.plot(comparison['AoAVsTime-time'], comparison['AoAVsTime-aoa'], '--', label=comparison['label'])
            if 'FlightPathAngleVsTime-time' in comparison:
                if 'FlightPathAngleVsTime-label' in comparison:
                    plt.plot(comparison['FlightPathAngleVsTime-time'], comparison['FlightPathAngleVsTime-fpa'], '--', label=comparison['FlightPathAngleVsTime-label'])
                else:
                    plt.plot(comparison['FlightPathAngleVsTime-time'], comparison['FlightPathAngleVsTime-fpa'], '--', label=comparison['label'])
    plt.title('Flight Path Angle and Angle of Attack vs Time')
    plt.ylabel('Angle (degrees)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    # Plot 3,2: Drag Coefficient and Atmospheric Density vs Time
    plt.subplot(3, 3, 8)
    ax5_1 = plt.gca()
    ax5_1.plot(times, drag_coeffs, 'b-', label='Drag Coefficient')
    ax5_1.set_title('Drag Coefficient, Atmospheric Pressure, Temperature, Density vs Time')
    ax5_1.set_ylabel('Drag Coefficient', color='b')
    ax5_1.tick_params(axis='y', labelcolor='b')
    ax5_1.grid(True)

    ax5_2 = ax5_1.twinx()
    ax5_2.plot(times, atm_pressures, 'g-', label='Atmospheric Pressure')
    ax5_2.set_ylabel('Pressure (Pa)', color='g')
    ax5_2.tick_params(axis='y', labelcolor='g')

    # chart atmospheric temperature on the left y-axis
    ax5_3 = ax5_1.twinx()
    ax5_3.spines['left'].set_position(('outward', 50))  # Move the right spine outward
    ax5_3.yaxis.tick_left()
    ax5_3.yaxis.set_label_position('left')
    ax5_3.spines['left'].set_visible(True)
    ax5_3.plot(times, atm_temperatures, 'r-', label='Atmospheric Temperature')
    ax5_3.set_ylabel('Temperature (K)', color='r')
    ax5_3.tick_params(axis='y', labelcolor='r')
    
    # Add legends for each y-axis
    lines, labels = ax5_1.get_legend_handles_labels()
    lines2, labels2 = ax5_2.get_legend_handles_labels()
    lines3, labels3 = ax5_3.get_legend_handles_labels()
    ax5_1.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='upper left')
    ax5_1.set_xlabel('Time (s)')

    # Plot 3,3: Parameters
    plt.subplot(3, 3, 9)
    plt.axis('off')
    
    plt.text(-0.15, 1.0, f"Parameters:", fontsize=10, fontweight='bold')
    
    plt.text(-0.15, 0.91, f"Mass: {params['mass']} kg", fontsize=10)
    plt.text(-0.15, 0.84, f"Area: {params['area']} m²", fontsize=10)
    plt.text(-0.15, 0.77, f"Ballistic Coefficient: {params['ballistic_coefficient']:.2f} kg/m²", fontsize=10)
    
    plt.text(-0.15, 0.67, f"Entry Altitude: {params['entry_altitude']} m", fontsize=10)
    plt.text(-0.15, 0.60, f"Entry Flight Path Angle: {params['entry_flight_path_angle']} degrees", fontsize=10)
    plt.text(-0.15, 0.53, f"Entry Velocity: {params['entry_velocity']} m/s", fontsize=10)
    
    plt.text(-0.15, 0.43, f"Time Step: {params['time_step']} second(s)", fontsize=10)
    plt.text(-0.15, 0.36, f"Max Time: {params['time_max']} seconds", fontsize=10)
    
    # Final values
    plt.text(0.4, 1, f"Terminal Values (At Impact):", fontsize=10, fontweight='bold')
    
    plt.text(0.4, 0.91, f"Final Altitude: {altitudes[-1]:.2f} m", fontsize=10)
    plt.text(0.4, 0.84, f"Final Downrange Distance: {downrange_dists[-1]:.2f} m", fontsize=10)
    
    # refactor all below to use -1 index instead of final
    plt.text(0.4, 0.77, f"Final Velocity: {velocities[-1]:.2f} m/s", fontsize=10)
    plt.text(0.4, 0.70, f"Final Horizontal Velocity: {v_xs[-1]:.2f} m/s", fontsize=10)
    plt.text(0.4, 0.63, f"Final Vertical Velocity: {v_ys[-1]:.2f} m/s", fontsize=10)
    
    plt.text(0.4, 0.53, f"Final Acceleration: {net_accs[-1]:.2f} m/s²", fontsize=10)
    plt.text(0.4, 0.46, f"Final Horizontal Acceleration: {a_xs[-1]:.2f} m/s²", fontsize=10)
    plt.text(0.4, 0.39, f"Final Vertical Acceleration: {a_ys[-1]:.2f} m/s²", fontsize=10)
    plt.text(0.4, 0.32, f"Final Drag Acceleration: {drag_accs[-1]:.2f} m/s²", fontsize=10)
    plt.text(0.4, 0.25, f"Final Gravity Acceleration: {grav_accs[-1]:.2f} m/s²", fontsize=10)
    
    plt.text(0.4, 0.15, f"Final Flight Path Angle: {flight_path_angles[-1]:.2f} degrees", fontsize=10)
    plt.text(0.4, 0.08, f"Final Time: {times[-1]:.2f} seconds", fontsize=10)
    
    plt.text(0.4, -0.02, f"Execution Time: {execution_time:.2f} seconds", fontsize=10)

    # Adjust layout
    plt.subplots_adjust(left=0.055, right=0.98, top=0.925, bottom=0.042, hspace=0.29, wspace=0.31)

    # Save the plot
    plt.savefig(filename)
    if show: plt.show()
    plt.close()

import backtest