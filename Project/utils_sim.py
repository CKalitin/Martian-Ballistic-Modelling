import numpy as np
import utils_data

def get_interpolated_drag_coefficient(vel):
    # vel = m/s
    if vel < utils_data.cd_vel_data[-1] or vel > utils_data.cd_vel_data[0]:
        raise ValueError(f"Velocity {vel} m/s is outside the data range [{utils_data.cd_vel_data[-1]}, {utils_data.cd_vel_data[0]}] m/s")
    
    # Since velocity_data is in descending order, reverse for np.interp
    return np.interp(vel, utils_data.cd_vel_data[::-1], utils_data.cd_drag_coeff_data[::-1])

def get_interpolated_lift_to_drag_ratio(aoa):
    # aoa = degrees
    if aoa < utils_data.ld_vel_data[0] or aoa > utils_data.ld_vel_data[-1]:
        raise ValueError(f"Angle of attack {aoa} degrees is outside the data range [{utils_data.ld_vel_data[-1]}, {utils_data.ld_vel_data[0]}] degrees")

    return np.interp(aoa, utils_data.ld_vel_data, utils_data.ld_ratio_data)

def get_numpy_aoa_list(aoa_list):
    # aoa_list = [(alt, aoa), ...], alt = meters, aoa = degrees
    out = np.array(aoa_list)
    # reverse the order of both columns to be in descending order
    out[:, 0] = out[:, 0][::-1]
    out[:, 1] = out[:, 1][::-1]
    return out

def get_interpolated_aoa(numpy_aoa_list, vel):
    return np.interp(vel, numpy_aoa_list[:,0], numpy_aoa_list[:,1])

def get_atmospheric_pressure(alt):
    # alt = meters, atmospheric pressure = Pa
    if (alt < utils_data.pressure_alt_data[0]):
        return 699*np.exp(-0.0000945*alt)

    return np.interp(alt, utils_data.pressure_alt_data, utils_data.pressure_data)

def get_temperature(alt):
    # alt = meters, temperature = K
    return np.interp(alt, utils_data.temp_alt_data, utils_data.temp_data)

# Alternative formula that gives default value of ~0.015, not using this one
def get_atmospheric_density(alt, pressure=None, temperature=None):
    # alt = meters, pressure = Pa, temperature = K, atmospheric density = kg/m^3
    if pressure is None:
        pressure = get_atmospheric_pressure(alt)
    if temperature is None:
        temperature = get_temperature(alt)
        
    return pressure / (192.1 * temperature)  # Ideal gas law: density = pressure / (R * T)

def get_gravity_acc(radial_distance):
    # radial_distance = distance from the center of Mars (m)
    return -utils_data.GRAVITATIONAL_CONSTANT * utils_data.MARS_MASS / radial_distance**2

def get_drag_acc(mass, vel_net, area, atm_density):
    # mass = kg, vel_net = m/s, area = m^2, atm_density = kg/m^3
    
    drag_coeff = get_interpolated_drag_coefficient(vel_net)
    
    return -0.5 * atm_density * vel_net**2 * drag_coeff * area / mass

def get_lift_acc(drag_acc, aoa):
    # drag_acc = m/s^2, flight_path_angle = degrees
    ld = get_interpolated_lift_to_drag_ratio(aoa)
    return -drag_acc * ld if ld != 0 else 0
