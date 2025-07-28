# Each time step

# Calculate atmospheric parameters (temp, rpessure, density)
# Calculate acceleration
# Calculate velocity
# Calculate flight path angle
# Update position vectors using flight path angle

# Make a struct-like object to hold all time series data
# Save to struct-like object for each iteration, don't make a calculate then copy lines, just do it in one line

import utils
import math
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass

# TODO:
# - Bank angle (angle between normal of the vehicle and normal of the surface)
# - Sideslip angle (horizontal plane angle, where angle of attack is vertical plane angle)

@dataclass
class SimData:
    """All simulation data as lists - ready for plotting."""
    t: list = None; alt: list = None; r_dist: list = None; ang_dist: list = None; ang_dist_rad: list = None; a_net: list = None
    a_rad: list = None; a_ang: list = None; a_grav: list = None; a_drag: list = None; a_lift: list = None
    v_net: list = None; v_rad: list = None; v_ang: list = None; fpa: list = None; aoa: list = None
    atm_p: list = None; atm_t: list = None; atm_rho: list = None; drag_coeff: list = None
    global_cartesian_pos_x: list = None; global_cartesian_pos_y: list = None; execution_time: list = None

    def __post_init__(self):
        for field in self.__dataclass_fields__:
            if getattr(self, field) is None:
                setattr(self, field, [])
    
    def add(self, *values):
        """Add timestep data - just pass 18 values in order."""
        for field, value in zip(self.__dataclass_fields__, values):
            getattr(self, field).append(value)
    
    def __getitem__(self, i):
        """Get timestep i as dict."""
        return {f: getattr(self, f)[i] for f in self.__dataclass_fields__}

def simulate(mass, area, entry_altitude, entry_flight_path_angle, entry_velocity, aoa_function=None, time_step=0.1, time_max=1000, verbose=False):
    """Simulate Mars entry trajectory.

    Args:
        mass (float): Mass of entry vehicle.
        entry_altitude (float): Initial altitude (m).
        entry_flight_path_angle (float): Initial flight path angle (degrees).
        entry_velocity (float): Initial velocity (m/s).
        aoa_function (callable, optional): Function returning angle of attack at time t.
        time_step (float, optional): Step size. Defaults to 0.1.
        time_max (int, optional): Terminate simulation once time step reaches this value. Defaults to 1000.
        verbose (bool, optional): Print progress. Defaults to False.
        
    Returns:
        list: Time series data of the simulation.
        dict: Impact values at the end of the simulation.
        dict: Parameters used in the simulation.
    """
    
    simulation_start_time = time.time()

    # If AoA is a single value, convert to list of lists of altitude vs. AoA (This is then interpolated to get AoA at any given altitude)
    if type(aoa_function) != list:
        aoa_function = [[entry_altitude, aoa_function], [0, aoa_function]]
    aoa_list = utils.get_numpy_aoa_list(aoa_function)
    
    altitude = entry_altitude
    radial_distance = entry_altitude + utils.MARS_RADIUS  # Distance from the center of Mars
    net_angular_distance_m = 0
    net_angular_distance_rad = 0 # radians
    
    vel_net_m = entry_velocity # m/s
    vel_rad_m = entry_velocity * math.sin(math.radians(entry_flight_path_angle)) # Radial velocity (m/s)
    vel_ang_m = entry_velocity * math.cos(math.radians(entry_flight_path_angle)) # Angular velocity (m/s)
    flight_path_angle = entry_flight_path_angle
    
    # Data collection
    data = SimData()
    
    t = 0
    while t < time_max and altitude > 0:
        # Calculate initial timestep conditions
        atm_pressure = utils.get_atmospheric_pressure(altitude)
        atm_temperature = utils.get_temperature(altitude)
        atm_density = utils.get_atmospheric_density(altitude, atm_pressure, atm_temperature)

        aoa = utils.get_interpolated_aoa(aoa_list, altitude)

        drag_coeff = utils.get_interpolated_drag_coefficient(vel_net_m)

        # Calculate acceleration
        a_grav = get_gravity_acc(radial_distance)
        a_drag = get_drag_acc(mass, vel_net_m, area, atm_density)
        a_lift = get_lift_acc(a_drag, aoa)

        print(f"alt: {altitude}, a_drag: {a_drag}, a_lift: {a_lift}, a_grav: {a_grav}, atm_pressure: {atm_pressure}, atm_density: {atm_density}")

        # Update velocities
        a_rad = a_grav + math.sin(math.radians(flight_path_angle)) * a_drag + math.sin(math.radians(flight_path_angle)) * a_lift
        a_ang = math.cos(math.radians(flight_path_angle)) * a_drag + math.cos(math.radians(flight_path_angle)) * a_lift
        a_net = math.sqrt(a_rad**2 + a_ang**2)
        
        vel_rad_m += a_rad * time_step
        vel_ang_m += a_ang * time_step
        vel_net_m = math.sqrt(vel_rad_m**2 + vel_ang_m**2)
        
        # Update flight path angle
        flight_path_angle = math.degrees(math.atan2(vel_rad_m, vel_ang_m)) # atan2(y, x) gives angle in radians, y = radial velocity, x = angular velocity

        # Update Positions
        radial_distance += vel_rad_m * time_step
        altitude = radial_distance - utils.MARS_RADIUS
        
        angular_distance_m = vel_ang_m * time_step
        angular_distance_rad = angular_distance_m / radial_distance # theta = s/r
        
        net_angular_distance_m = net_angular_distance_m + angular_distance_m
        net_angular_distance_rad = net_angular_distance_rad + angular_distance_rad
        
        global_cartesian_pos_x = radial_distance * math.sin(net_angular_distance_rad) / 1000 # Convert to km
        global_cartesian_pos_y = radial_distance * math.cos(net_angular_distance_rad) / 1000
        
        data.add(t, altitude, radial_distance, net_angular_distance_m, net_angular_distance_rad, a_net, a_rad, a_ang, a_grav, a_drag, a_lift, vel_net_m, vel_rad_m, vel_ang_m, flight_path_angle, aoa, atm_pressure, atm_temperature, atm_density, drag_coeff, global_cartesian_pos_x, global_cartesian_pos_y, time.time() - simulation_start_time)
        if verbose: print(f"{data[-1]}\n")
        
        # Rotate velocity vector counterclockwise so velocity is consistent in a global cartesian coordinate system
        temp_vel_rad_m = vel_rad_m * math.cos(angular_distance_rad) + vel_ang_m * math.sin(angular_distance_rad)
        temp_vel_ang_m = -vel_rad_m * math.sin(angular_distance_rad) + vel_ang_m * math.cos(angular_distance_rad)
        vel_rad_m, vel_ang_m = temp_vel_rad_m, temp_vel_ang_m
        
        t += time_step

    parameters = {'mass': mass, 'area': area, 'ballistic_coefficient': mass/area, 'entry_altitude': entry_altitude, 'entry_flight_path_angle': entry_flight_path_angle, 'entry_velocity': entry_velocity, 'time_step': time_step, 'time_max': time_max}
    
    return data, parameters

GRAVITATIONAL_CONSTANT = 6.67430e-11  # m^3 kg^-1 s^-2
MARS_MASS = 6.4171e23  # Mars mass in kg

def get_gravity_acc(radial_distance):
    # radial_distance = distance from the center of Mars (m)
    return -GRAVITATIONAL_CONSTANT * MARS_MASS / radial_distance**2

def get_drag_acc(mass, vel_net, area, atm_density):
    # mass = kg, vel_net = m/s, area = m^2, atm_density = kg/m^3
    
    drag_coeff = utils.get_interpolated_drag_coefficient(vel_net)
    
    return -0.5 * atm_density * vel_net**2 * drag_coeff * area / mass

def get_lift_acc(drag_acc, aoa):
    # drag_acc = m/s^2, flight_path_angle = degrees
    ld = utils.get_interpolated_lift_to_drag_ratio(aoa)
    return -drag_acc * ld if ld != 0 else 0

def plot(data, parameters, title="Mars Entry Simulation", file_name="mars_entry_simulation.png", show=False, comparisons=None):
    # Comparisions is a list of tuples (velocity, altitude, label), MAKE SURE ITS A LIST, NOT JUST A TUPLE, USE THE SQUARE BRACKETS
    if comparisons is None:
        comparisons = []
        
    plt.figure(figsize=(19.20, 10.80), dpi=100)
    plt.suptitle(title, fontsize=16) # Supertitle
    plt.gcf().text(0.01, 0.965, f"Christopher Kalitin 2025", fontsize=12)
    plt.axis('off')
    
    # Note we're assuming horizontal velocity = angular velocity, and vertical velocity = radial velocity, which at any given instant relative to the surface should be true
    
    sub_plot((3,3,1), "Altitude vs Time", "Time (s)", "Altitude (m)", data.t, [data.alt], ["Simulation"], comparisons, 'AltVsTime-time', 'AltVsTime-alt')
    sub_plot((3,3,2), "Altitude vs Velocity", "Velocity (m/s)", "Altitude (m)", data.v_net, [data.alt], ["Simulation"], comparisons, 'AltVsVel-vel', 'AltVsVel-alt')
    sub_plot((3,3,3), "Global Cartesian Position", "X Position (km)", "Y Position (km)", data.global_cartesian_pos_x, [data.global_cartesian_pos_y], ["Global Cartesian Position"], comparisons, ['body_points_x'], ['body_points_y'], equal_aspect=True)
    #sub_plot((3,3,3), "Altitude vs Downrange Distance (Angular)", "Downrange Distance (m)", "Altitude (m)", data.ang_dist, [data.alt], ["Simulation"], comparisons, 'AltVsDownrangeDist-dist', 'AltVsDownrangeDist-alt')
    sub_plot((3,3,4), "Velocities vs Time", "Time (s)", "Velocity (m/s)", data.t, [data.v_net, data.v_ang, data.v_rad], ["Net Velocity", "Horizontal Velocity", "Vertical Velocity"], comparisons, 'VelVsTime-time', ['VelVsTime-vel', 'HVelVsTime-vel', 'VVelVsTime-vel'])
    sub_plot((3,3,5), "Acceleration vs Time", "Time (s)", "Acceleration (m/s²)", data.t, [data.a_net, data.a_ang, data.a_rad], ["Net Acceleration", "Horizontal Acceleration", "Vertical Acceleration"])
    sub_plot((3,3,6), "Drag, Lift, Gravity Acceleration vs Time", "Time (s)", "Acceleration (m/s²)", data.t, [data.a_drag, data.a_lift, data.a_grav], ["Drag Acceleration", "Lift Acceleration", "Gravity Acceleration"])
    sub_plot((3,3,7), "Flight Path Angle and Angle of Attack vs Time", "Time (s)", "Angle (degrees)", data.t, [data.fpa, data.aoa], ["Flight Path Angle", "Angle of Attack"], comparisons, ['AoAVsTime-time', 'FlightPathAngleVsTime-time'], ['AoAVsTime-aoa', 'FlightPathAngleVsTime-fpa'], 'FlightPathAngleVsTime-label')
    sub_plot_atmosphere((3,3,8), data)
    sub_plot_text((3,3,9), parameters, data)
    
    plt.subplots_adjust(left=0.055, right=0.98, top=0.925, bottom=0.042, hspace=0.29, wspace=0.31)

    plt.savefig(file_name)
    if show: plt.show()
    plt.close()

def sub_plot(position, title, x_label, y_label, x_data, y_data_list, series_labels, comparisons=[], comparison_x_key=None, comparison_y_keys=None, comparison_label_field='label', equal_aspect = False):
    """
    Creates a subplot and plots multiple data series, with optional comparison series.
        position (tuple): A tuple (nrows, ncols, index) specifying the subplot position.
        title (str): The title of the subplot.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        x_data (array-like): The data for the x-axis, shared by all main series.
        y_data_list (list of array-like): List of y-axis data arrays, one for each main series.
        series_labels (list of str): List of labels for each main series.
        comparisons (list of dict, optional): List of comparison data dictionaries. Each dictionary should contain keys for x and y data and label.
        comparison_x_key (list of str, optional): List of keys to extract x-axis data from each comparison dictionary.
        comparison_y_keys (list of str, optional): List of keys to extract y-axis data from each comparison dictionary.
        comparison_label_field (str, optional): The key in the comparison dictionaries to use for the label. Defaults to 'label'.
        equal_aspect (bool, optional): Whether to set equal aspect ratio for the subplot. Defaults to False.
    Returns:
        None: The function creates a subplot and plots the provided data.
    Notes:
        - Adds a legend if more than one series is plotted.
        - Comparison series are plotted with dashed lines.
        - array-like means any structure that can be converted to a numpy array, such as lists or numpy arrays.
    """

    zorder = 999
    
    plt.subplot(position[0], position[1], position[2])

    for y_data, label in zip(y_data_list, series_labels):
        plt.plot(x_data, y_data, label=label, zorder=zorder)
        zorder -= 1
        
    for comparison in comparisons:
        for x_key, y_key in zip(comparison_x_key, comparison_y_keys):
            if x_key in comparison and y_key in comparison:
                label_field = comparison_label_field if comparison_label_field in comparison else 'label' # I FUCKING LOVE PYTHON
                plt.plot(comparison[x_key], comparison[y_key], '--', label=comparison[label_field], zorder=zorder)
                zorder -= 1
                
    if equal_aspect:
        x_range = max(x_data) - min(x_data) # x range of the data
        y_range = max(y_data_list[0]) - min(y_data_list[0])
        fig_aspect = plt.gcf().get_size_inches()[0] / position[0] / (plt.gcf().get_size_inches()[1] / position[1]) # On screen x to y pixel ratio
        if x_range / y_range > fig_aspect: # If the x range is larger than the y range, adjust y limits
            y_center = sum(y_data_list[0]) / len(y_data_list[0])
            plt.ylim(y_center - x_range / fig_aspect / 2, y_center + x_range / fig_aspect / 2)
        else: # If the y range is larger than the x range, adjust x limits
            x_center = sum(x_data) / len(x_data)
            plt.xlim(x_center - y_range * fig_aspect / 2, x_center + y_range * fig_aspect / 2)
        plt.gca().set_aspect('equal', adjustable='datalim')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if len(y_data_list) + len(comparisons) > 1: plt.legend()
    plt.grid(True)

def sub_plot_atmosphere(position, data):
    plt.subplot(position[0], position[1], position[2])
    ax5_1 = plt.gca()
    ax5_1.plot(data.t, data.drag_coeff, 'b-', label='Drag Coefficient')
    ax5_1.set_title('Drag Coefficient, Atmospheric Pressure, Temperature, Density vs Time')
    ax5_1.set_ylabel('Drag Coefficient', color='b')
    ax5_1.tick_params(axis='y', labelcolor='b')
    ax5_1.grid(True)

    ax5_2 = ax5_1.twinx()
    ax5_2.plot(data.t, data.atm_p, 'g-', label='Atmospheric Pressure')
    ax5_2.set_ylabel('Pressure (Pa)', color='g')
    ax5_2.tick_params(axis='y', labelcolor='g')

    # chart atmospheric temperature on the left y-axis
    ax5_3 = ax5_1.twinx()
    ax5_3.spines['left'].set_position(('outward', 55))  # Move the right spine outward
    ax5_3.yaxis.tick_left()
    ax5_3.yaxis.set_label_position('left')
    ax5_3.spines['left'].set_visible(True)
    ax5_3.plot(data.t, data.atm_t, 'r-', label='Atmospheric Temperature')
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

def sub_plot_text(position, parameters, data):
    """
    Creates a subplot and plots multiple data series with text annotations.
    """
    
    plt.subplot(position[0], position[1], position[2])
    plt.axis('off')

    plt.text(-0.15, 1.0, f"Parameters:", fontsize=10, fontweight='bold')
    
    plt.text(-0.15, 0.91, f"Mass: {parameters['mass']} kg", fontsize=10)
    plt.text(-0.15, 0.84, f"Area: {parameters['area']} m²", fontsize=10)
    plt.text(-0.15, 0.77, f"Ballistic Coefficient: {parameters['ballistic_coefficient']:.2f} kg/m²", fontsize=10)

    plt.text(-0.15, 0.67, f"Entry Altitude: {parameters['entry_altitude']} m", fontsize=10)
    plt.text(-0.15, 0.60, f"Entry Flight Path Angle: {parameters['entry_flight_path_angle']} degrees", fontsize=10)
    plt.text(-0.15, 0.53, f"Entry Velocity: {parameters['entry_velocity']} m/s", fontsize=10)

    plt.text(-0.15, 0.43, f"Time Step: {parameters['time_step']} second(s)", fontsize=10)
    plt.text(-0.15, 0.36, f"Max Time: {parameters['time_max']} seconds", fontsize=10)

    # Final values
    plt.text(0.4, 1, f"Terminal Values (At Impact):", fontsize=10, fontweight='bold')

    plt.text(0.4, 0.91, f"Final Altitude: {data.alt[-1]:.2f} m", fontsize=10)
    plt.text(0.4, 0.84, f"Final Downrange Distance: {data.ang_dist[-1]:.2f} m", fontsize=10)
    plt.text(0.4, 0.77, f"Final Angular Distance (Deg): {math.degrees(data.ang_dist_rad[-1]):.2f} degrees", fontsize=10)

    plt.text(0.4, 0.67, f"Final Velocity: {data.v_net[-1]:.2f} m/s", fontsize=10)
    plt.text(0.4, 0.60, f"Final Horizontal Velocity: {data.v_ang[-1]:.2f} m/s", fontsize=10)
    plt.text(0.4, 0.53, f"Final Vertical Velocity: {data.v_rad[-1]:.2f} m/s", fontsize=10)
    
    plt.text(0.4, 0.43, f"Final Acceleration: {data.a_net[-1]:.2f} m/s²", fontsize=10)
    plt.text(0.4, 0.36, f"Final Horizontal Acceleration: {data.a_ang[-1]:.2f} m/s²", fontsize=10)
    plt.text(0.4, 0.29, f"Final Vertical Acceleration: {data.a_rad[-1]:.2f} m/s²", fontsize=10)
    plt.text(0.4, 0.22, f"Final Drag Acceleration: {data.a_drag[-1]:.2f} m/s²", fontsize=10)
    plt.text(0.4, 0.15, f"Final Gravity Acceleration: {data.a_grav[-1]:.2f} m/s²", fontsize=10)

    plt.text(0.4, 0.05, f"Final Flight Path Angle: {data.fpa[-1]:.2f} degrees", fontsize=10)
    plt.text(0.4, -0.02, f"Final Time: {data.t[-1]:.2f} seconds", fontsize=10)

    plt.text(0.4, -0.12, f"Execution Time: {data.execution_time[-1]:.2f} seconds", fontsize=10)  

comparison = { 'body_points_x': utils.mars_circumference_points_km_x, 'body_points_y': utils.mars_circumference_points_km_y, 'label': 'Mars' }

data, parameters = simulate(
    mass=1000, 
    area=10, 
    entry_altitude=1000000, 
    entry_flight_path_angle=0, 
    entry_velocity=3500, 
    aoa_function=0,
    time_step=10, 
    time_max=100000, 
    verbose=False
)

def remove_comparison_body_points_out_of_range(comparison, data):
    """
    Removes points from comparison['body_points_x'] and comparison['body_points_y'] that are out of the specified range.
    The lists are modified in place.
    
    Find the farthest points (in +/- x & y), if any body points are out of range, remove them
    If a point in the x list is out of range, remove the corresponding point in the y list
    """
    
    margin = 50 # km
    
    x_min = min(data.global_cartesian_pos_x) - margin
    x_max = max(data.global_cartesian_pos_x) + margin
    y_min = min(data.global_cartesian_pos_y) - margin
    y_max = max(data.global_cartesian_pos_y) + margin
    
    x_points = comparison['body_points_x']
    y_points = comparison['body_points_y']
    filtered_x = []
    filtered_y = []
    for x, y in zip(x_points, y_points):
        if x_min <= x <= x_max and y_min <= y <= y_max:
            filtered_x.append(x)
            filtered_y.append(y)
    comparison['body_points_x'] = filtered_x
    comparison['body_points_y'] = filtered_y

remove_comparison_body_points_out_of_range(comparison, data)

# remove last 3 from x and y, yea im just hardcoding this in, it makes it loop back and puts an ugly line across the graph
comparison['body_points_x'] = comparison['body_points_x'][:-3]
comparison['body_points_y'] = comparison['body_points_y'][:-3]

plot(data, parameters, show=True, file_name="test.png", comparisons=[comparison])

