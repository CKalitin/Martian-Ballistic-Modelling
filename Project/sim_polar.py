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
class SimStep:
    """Single simulation timestep data - clean and compact."""
    t: float; alt: float; r_dist: float; ang_dist: float; a_rad: float; a_ang: float
    a_grav: float; a_drag: float; a_lift: float; v_net: float; v_rad: float; v_ang: float
    fpa: float; aoa: float; p_atm: float; t_atm: float; rho_atm: float; sim_time: float

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
    data = []
    
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
        
        # Save data in one line
        data.append(SimStep(t, altitude, radial_distance, angular_distance_m, a_rad, a_ang, a_grav, a_drag, a_lift, vel_net_m, vel_rad_m, vel_ang_m, flight_path_angle, aoa, atm_pressure, atm_temperature, atm_density, time.time() - simulation_start_time))

        if verbose:
            print(f"{data[-1]}\n")

        t += time_step

    impact_values = data[-1]
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

def sub_plot(position, title, x_label, y_label, series_x, series_y_lists, series_labels, comparisons=None, comparison_series_x_key=None, comparison_series_y_keys=None, comparison_series_y_label_keys=None):
    """Create a subplot for the given data.

    Args:
        title (_type_): _description_
        x_label (_type_): _description_
        y_label (_type_): _description_
        series_x (_type_): _description_
        series_y_lists (_type_): _description_
        series_labels (_type_): _description_
        comparisons (_type_): _description_
        comparison_series_x_titlecomparison_series_y_titles (_type_): _description_
    """
    
    z_order = 999
    
    plt.subplot(position[0], position[1], position[2])
    
    for series_y_list, label in zip(series_y_lists, series_labels):
        plt.plot(series_x, series_y_list, label=label, z_order=z_order)
        z_order -= 1

    if comparisons is not None:
        for comparison, x_key, y_key, y_label_key in zip(comparisons, comparison_series_x_key, comparison_series_y_keys, comparison_series_y_label_keys):
            plt.plot(comparisons[x_key], comparisons[y_key], '--', label=comparison[y_label_key])

    plt.title(plt.title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if len(series_y_lists) + len(comparisons) > 1:
        plt.legend()
    plt.grid(True)
    
simulate(
    mass=1000, 
    area=10, 
    entry_altitude=100000, 
    entry_flight_path_angle=-12, 
    entry_velocity=5000, 
    aoa_function=5,
    time_step=1, 
    time_max=10, 
    verbose=True
)