import numpy as np

# Coefficient of drag data, drag coefficient vs. velocity (m/s)
cd_drag_coeff_data = np.array([
    1.73, 1.72, 1.71, 1.71, 1.72, 1.73, 1.73, 1.73, 1.73, 1.73,
    1.73, 1.73, 1.73, 1.72, 1.71, 1.70, 1.69, 1.68, 1.67, 1.66,
    1.65, 1.65, 1.64, 1.64, 1.63, 1.62, 1.61, 1.60, 1.59, 1.58,
    1.58, 1.58, 1.58, 1.57, 1.57, 1.57, 1.57, 1.54, 1.42, 1.30,
    1.22, 1.16, 1.10, 1.04, 1.00
])

cd_vel_data = np.array([
    100000, 5890, 5880, 5850, 5800, 5590, 5400, 5200, 5000, 4790,
    4590, 4400, 4190, 3990, 3790, 3600, 3400, 3200, 3000, 2800,
    2600, 2400, 2190, 1990, 1790, 1590, 1400, 1200, 1000, 800,
    620, 540, 490, 460, 430, 400, 370, 310, 260, 220,
    190, 180, 150, 100, 0, 
])

# Lift to drag ratio data, lift to drag ratio vs. AoA (degrees)
ld_ratio_data = np.array([
    -1.5,-0.265,-0.175,-0.095,-0.065,-0.035,0,0.035,0.065,0.095,0.175,0.265,1.5
])

ld_vel_data = np.array([
    -90,-16,-11,-6,-4,-2,0,2,4,6,11,16,90
])

# Atmospheric pressure data, pressure vs. altitude (m)
pressure_data = np.array([
    23.257310206, 9.565493821, 2.925754907, 0.963657775, 0.254181943, 0.097081899,
    0.031975928, 0.009082373, 0.002991467, 0.000985301, 0.000324529, 0.000092179,
    0.000035207, 0.000010000, 0.000002840, 0.000000936, 0.000000308, 0.000000094,
    0.000000036, 0.000000009, 0.000000003, 0.000000001, 0.000000000, 0.0
])

pressure_alt_data = np.array([
    35950, 42373, 52754, 60169, 70551, 77966, 85381, 94280, 100212, 107627,
    115042, 125424, 132839, 147669, 161017, 174364, 189195, 208475, 226271, 258898,
    303390, 361229, 509534, 1000000000
])

# Temperature data, temperature (K) vs. altitude (m)
temp_data = np.array([
    222, 222, 209, 202, 189, 171, 156, 150, 148, 144, 132,
    130, 134, 136, 133, 129, 128, 135, 149, 176, 205,
    219, 232, 240, 245, 251, 252, 252
])

temp_alt_data = np.array([
    -10000000, 0, 3849, 10067, 17174, 27833, 37308, 44415, 56555, 63069,
    72840, 80242, 85572, 92678, 99785, 104522, 110148, 113109, 116958, 122288,
    130283, 135612, 143903, 152490, 161669, 180915, 203122, 100000000
])

def get_interpolated_drag_coefficient(vel):
    # vel = m/s
    if vel < cd_vel_data[-1] or vel > cd_vel_data[0]:
        raise ValueError(f"Velocity {vel} m/s is outside the data range [{cd_vel_data[-1]}, {cd_vel_data[0]}] m/s")
    
    # Since velocity_data is in descending order, reverse for np.interp
    return np.interp(vel, cd_vel_data[::-1], cd_drag_coeff_data[::-1])

def get_interpolated_lift_to_drag_ratio(aoa):
    # aoa = degrees
    if aoa < ld_vel_data[0] or aoa > ld_vel_data[-1]:
        raise ValueError(f"Angle of attack {aoa} degrees is outside the data range [{ld_vel_data[-1]}, {ld_vel_data[0]}] degrees")
    
    return np.interp(aoa, ld_vel_data, ld_ratio_data)

def get_numpy_aoa_list(aoa_list):
    # aoa_list = [(alt, aoa), ...], alt = meters, aoa = degrees
    out = np.array(aoa_list)
    # reverse the order of both columns to be in descending order
    out[:, 0] = out[:, 0][::-1]
    out[:, 1] = out[:, 1][::-1]
    return out

def get_interpolated_aoa(numpy_aoa_list, alt):
    return np.interp(alt, numpy_aoa_list[:,0], numpy_aoa_list[:,1])

def get_atmospheric_pressure(alt):
    # alt = meters, atmospheric pressure = Pa
    if (alt < pressure_alt_data[0]):
        return 699*np.exp(-0.0000945*alt)

    return np.interp(alt, pressure_alt_data, pressure_data)

def get_temperature(alt):
    # alt = meters, temperature = K
    return np.interp(alt, temp_alt_data, temp_data)

def get_atmospheric_density(alt):
    # alt = meters, atmospheric density = kg/m^3
    surface_density = 0.0215 # kg/m^3 at sea level on Mars  
    scale_height = 10000  # Scale height in meters
    return surface_density * np.exp(-alt / scale_height)

# Alternative formula that gives default value of ~0.015, not using this one
def get_atmospheric_density_other(alt, pressure=None, temperature=None):
    # alt = meters, pressure = Pa, temperature = K, atmospheric density = kg/m^3
    if pressure is None:
        pressure = get_atmospheric_pressure(alt)
    if temperature is None:
        temperature = get_temperature(alt)
        
    return pressure / (192.1 * temperature)  # Ideal gas law: density = pressure / (R * T)

def get_gravity_acc(alt):
    # alt = meters, gravity = m/s^2
    
    gravitational_constant = 6.67430e-11  # m^3 kg^-1 s^-2
    mars_radius = 3396200  # Mars radius in meters
    mars_mass = 6.4171e23  # Mars mass in kg
    
    distance = mars_radius + alt  # Distance from the center of Mars
    
    return -gravitational_constant * (mars_mass) / (distance ** 2)

def get_drag_acc(alt, vel, area, mass):
    # alt = meters, vel = m/s, area = m^2, mass = kg
    return -1/mass * 0.5 * get_atmospheric_density(alt) * get_interpolated_drag_coefficient(vel) * area * vel**2

def get_drag_acc(mass, vel, area, drag_coeff, atm_density):
    # atm_density = kg/m^3, vel = m/s, area = m^2, drag_coeff = dimensionless
    return -1/mass * 0.5 * atm_density * drag_coeff * area * vel**2

def get_lift_acc(drag_acc, lift_to_drag_ratio):
    # drag_acc = m/s^2, lift_to_drag_ratio = dimensionless
    if lift_to_drag_ratio == 0:
        return 0
    return -drag_acc * lift_to_drag_ratio