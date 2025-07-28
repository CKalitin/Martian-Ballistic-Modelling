# Each time step

# Calculate atmospheric parameters (temp, rpessure, density)
# Calculate acceleration
# Calculate velocity
# Calculate flight path angle
# Update position vectors using flight path angle

# Make a struct-like object to hold all time series data
# Save to struct-like object for each iteration, don't make a calculate then copy lines, just do it in one line

import matplotlib.pyplot as plt
import math
import time
from dataclasses import dataclass

import utils_sim
import utils_chart
import utils_data

# TODO:
# - Bank angle (angle between normal of the vehicle and normal of the surface)

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
    aoa_list = utils_sim.get_numpy_aoa_list(aoa_function)
    
    altitude = entry_altitude
    radial_distance = entry_altitude + utils_data.MARS_RADIUS  # Distance from the center of Mars
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
        atm_pressure = utils_sim.get_atmospheric_pressure(altitude)
        atm_temperature = utils_sim.get_temperature(altitude)
        atm_density = utils_sim.get_atmospheric_density(altitude, atm_pressure, atm_temperature)

        aoa = utils_sim.get_interpolated_aoa(aoa_list, altitude)

        drag_coeff = utils_sim.get_interpolated_drag_coefficient(vel_net_m)

        # Calculate acceleration
        a_grav = utils_sim.get_gravity_acc(radial_distance)
        a_drag = utils_sim.get_drag_acc(mass, vel_net_m, area, atm_density)
        a_lift = utils_sim.get_lift_acc(a_drag, aoa)

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
        altitude = radial_distance - utils_data.MARS_RADIUS
        
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

def plot(data, parameters, title="Mars Entry Simulation", file_name="mars_entry_simulation.png", show=False, comparisons=None):
    # Comparisions is a list of tuples (velocity, altitude, label), MAKE SURE ITS A LIST, NOT JUST A TUPLE, USE THE SQUARE BRACKETS
    if comparisons is None:
        comparisons = []
        
    plt.figure(figsize=(19.20, 10.80), dpi=100)
    plt.suptitle(title, fontsize=16) # Supertitle
    plt.gcf().text(0.01, 0.965, f"Christopher Kalitin 2025", fontsize=12)
    plt.axis('off')
    
    # Add Mars body for global cartesian position chart
    comparisons.append({'body_points_x': utils_data.mars_circumference_points_km_x, 'body_points_y': utils_data.mars_circumference_points_km_y, 'label': 'Mars'})
    utils_chart.remove_comparison_body_points_out_of_range(comparisons[-1], data)

    # remove last 3 from x and y, yea im just hardcoding this in, it makes it loop back and puts an ugly line across the graph
    comparisons[-1]['body_points_x'] = comparisons[-1]['body_points_x'][:-3]
    comparisons[-1]['body_points_y'] = comparisons[-1]['body_points_y'][:-3]

    # Note we're assuming horizontal velocity = angular velocity, and vertical velocity = radial velocity, which at any given instant relative to the surface should be true

    utils_chart.sub_plot((3,3,1), "Altitude vs Time", "Time (s)", "Altitude (m)", data.t, [data.alt], ["Simulation"], comparisons, ['AltVsTime-time'], ['AltVsTime-alt'])
    utils_chart.sub_plot((3,3,2), "Altitude vs Velocity", "Velocity (m/s)", "Altitude (m)", data.v_net, [data.alt], ["Simulation"], comparisons, ['AltVsVel-vel'], ['AltVsVel-alt'])
    utils_chart.sub_plot((3,3,3), "Global Cartesian Position", "X Position (km)", "Y Position (km)", data.global_cartesian_pos_x, [data.global_cartesian_pos_y], ["Global Cartesian Position"], comparisons, ['body_points_x', 'global_cartesian_pos_x'], ['body_points_y', 'global_cartesian_pos_y'], equal_aspect=True)
    #utils_chart.sub_plot((3,3,3), "Altitude vs Downrange Distance (Angular)", "Downrange Distance (m)", "Altitude (m)", data.ang_dist, [data.alt], ["Simulation"], comparisons, 'AltVsDownrangeDist-dist', 'AltVsDownrangeDist-alt')
    utils_chart.sub_plot((3,3,4), "Velocities vs Time", "Time (s)", "Velocity (m/s)", data.t, [data.v_net, data.v_ang, data.v_rad], ["Net Velocity", "Horizontal Velocity", "Vertical Velocity"], comparisons, ['VelVsTime-time', 'HVelVsTime-time', 'VVelVsTime-time'], ['VelVsTime-vel', 'HVelVsTime-vel', 'VVelVsTime-vel'])
    utils_chart.sub_plot((3,3,5), "Acceleration vs Time", "Time (s)", "Acceleration (m/s²)", data.t, [data.a_net, data.a_ang, data.a_rad], ["Net Acceleration", "Horizontal Acceleration", "Vertical Acceleration"])
    utils_chart.sub_plot((3,3,6), "Drag, Lift, Gravity Acceleration vs Time", "Time (s)", "Acceleration (m/s²)", data.t, [data.a_drag, data.a_lift, data.a_grav], ["Drag Acceleration", "Lift Acceleration", "Gravity Acceleration"])
    utils_chart.sub_plot((3,3,7), "Flight Path Angle and Angle of Attack vs Time", "Time (s)", "Angle (degrees)", data.t, [data.fpa, data.aoa], ["Flight Path Angle", "Angle of Attack"], comparisons, ['AoAVsTime-time', 'FlightPathAngleVsTime-time'], ['AoAVsTime-aoa', 'FlightPathAngleVsTime-fpa'], 'FlightPathAngleVsTime-label')
    utils_chart.sub_plot_atmosphere((3,3,8), data)
    utils_chart.sub_plot_text((3,3,9), parameters, data)

    plt.subplots_adjust(left=0.055, right=0.98, top=0.925, bottom=0.042, hspace=0.29, wspace=0.31)

    plt.savefig(file_name)
    if show: plt.show()
    plt.close()
