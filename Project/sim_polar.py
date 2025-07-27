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
    t: list = None; alt: list = None; r_dist: list = None; ang_dist: list = None
    a_rad: list = None; a_ang: list = None; a_grav: list = None; a_drag: list = None; a_lift: list = None
    v_net: list = None; v_rad: list = None; v_ang: list = None; fpa: list = None; aoa: list = None
    p_atm: list = None; t_atm: list = None; rho_atm: list = None; sim_time: list = None

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
    angular_distance_m = 0
    
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

        # Calculate acceleration
        a_grav = get_gravity_acc(radial_distance)
        a_drag = get_drag_acc(mass, vel_net_m, area, atm_density)
        a_lift = get_lift_acc(a_drag, aoa)

        # Update velocities
        a_rad = a_grav + math.sin(math.radians(flight_path_angle)) * a_drag + math.sin(math.radians(flight_path_angle)) * a_lift
        a_ang = math.cos(math.radians(flight_path_angle)) * a_drag + math.cos(math.radians(flight_path_angle)) * a_lift
        
        vel_rad_m += a_rad * time_step
        vel_ang_m += a_ang * time_step
        vel_net_m = math.sqrt(vel_rad_m**2 + vel_ang_m**2)
        
        # Update flight path angle
        flight_path_angle = math.degrees(math.atan2(vel_rad_m, vel_ang_m)) # atan2(y, x) gives angle in radians, y = radial velocity, x = angular velocity

        # Update Positions
        radial_distance += vel_rad_m * time_step
        angular_distance_m += vel_ang_m * time_step
        altitude = radial_distance - utils.MARS_RADIUS
        
        data.add(t, altitude, radial_distance, angular_distance_m, a_rad, a_ang, a_grav, a_drag, a_lift, vel_net_m, vel_rad_m, vel_ang_m, flight_path_angle, aoa, atm_pressure, atm_temperature, atm_density, time.time() - simulation_start_time)
        if verbose: print(f"{data[-1]}\n")

        t += time_step

    impact_values = data[-1]  # Last timestep as dict
    parameters = {'mass': mass, 'area': area, 'entry_altitude': entry_altitude, 'entry_flight_path_angle': entry_flight_path_angle, 'entry_velocity': entry_velocity, 'time_step': time_step, 'time_max': time_max}
    
    return data, impact_values, parameters

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

def plot(data, parameters, impact_values, title="Mars Entry Simulation", file_name="mars_entry_simulation.png", show=False, comparisons=None):
    # Comparisions is a list of tuples (velocity, altitude, label), MAKE SURE ITS A LIST, NOT JUST A TUPLE, USE THE SQUARE BRACKETS
    if comparisons is None:
        comparisons = []
        
    plt.figure(figsize=(19.20, 10.80), dpi=100)
    plt.suptitle(title, fontsize=16) # Supertitle
    plt.gcf().text(0.01, 0.965, f"Christopher Kalitin 2025", fontsize=12)
    plt.axis('off')
    
    sub_plot((3,3,1), "Altitude vs Time", "Time (s)", "Altitude (m)", data.v_net, data.alt, "Simulation")
    
    plt.subplots_adjust(left=0.055, right=0.98, top=0.925, bottom=0.042, hspace=0.29, wspace=0.31)

    plt.savefig(file_name)
    if show: plt.show()
    plt.close()

def sub_plot(position, title, x_label, y_label, x_data, y_data_list, series_labels, comparisons=[], comparison_x_key=None, comparison_y_keys=None, comparison_y_label_keys=None):
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
        comparison_y_label_keys (list of str, optional): List of keys to extract label for each comparison series.
    Returns:
        None: The function creates a subplot and plots the provided data.
    Notes:
        - Adds a legend if more than one series is plotted.
        - Comparison series are plotted with dashed lines.
        - array-like means any structure that can be converted to a numpy array, such as lists or numpy arrays.
    """

    # If these are lists of strs and not lists of lists of strs, convert
    if (y_data_list is not None and type(y_data_list[0]) != list): y_data_list = [y_data_list]
    if (series_labels is not None and type(series_labels[0]) != list): series_labels = [series_labels]
    if (comparison_y_keys is not None and type(comparison_y_keys[0]) != list): comparison_y_keys = [comparison_y_keys]
    if (comparison_y_label_keys is not None and type(comparison_y_label_keys[0]) != list): comparison_y_label_keys = [comparison_y_label_keys]

    zorder = 999
    
    plt.subplot(position[0], position[1], position[2])

    for y_data, label in zip(y_data_list, series_labels):
        plt.plot(x_data, y_data, label=label, zorder=zorder)
        zorder -= 1
        
    for comparison in comparisons:
        for x_key, y_key, y_label_key in zip(comparison_x_key, comparison_y_keys, comparison_y_label_keys):
            if x_key in comparison and y_key in comparison and y_label_key in comparison:
                plt.plot(comparison[x_key], comparison[y_key], '--', label=comparison[y_label_key])

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if len(y_data_list) + len(comparisons) > 1: plt.legend()
    plt.grid(True)
    
def text_sub_plot(position, parameters, impact_values):
    """
    Creates a subplot and plots multiple data series with text annotations.
    """
    
    if type(y_data_list) != list: y_data_list = [y_data_list]
    if type(series_labels) != list: series_labels = [series_labels]
    if type(comparison_y_keys) != list: comparison_y_keys = [comparison_y_keys]
    if type(comparison_y_label_keys) != list: comparison_y_label_keys = [comparison_y_label_keys]
    
    z_order = 999
    
    plt.subplot(position[0], position[1], position[2])

    for y_data, label in zip(y_data_list, series_labels):
        pass    

data, impact_values, parameters = simulate(
    mass=1000, 
    area=10, 
    entry_altitude=100000, 
    entry_flight_path_angle=-12, 
    entry_velocity=5000, 
    aoa_function=5,
    time_step=1, 
    time_max=1000, 
    verbose=True
)

plot(data, parameters, impact_values, show=True, file_name="test.png")