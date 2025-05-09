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

# Atmospheric pressure data, pressure vs. altitude (m)
pressure_data = np.array([
    23.257310206, 9.565493821, 2.925754907, 0.963657775, 0.254181943, 0.097081899,
    0.031975928, 0.009082373, 0.002991467, 0.000985301, 0.000324529, 0.000092179,
    0.000035207, 0.000010000, 0.000002840, 0.000000936, 0.000000308, 0.000000094,
    0.000000036, 0.000000009, 0.000000003, 0.000000001, 0.000000000, 0.0
])

altitude_data = np.array([
    34958, 42373, 52754, 60169, 70551, 77966, 85381, 94280, 100212, 107627,
    115042, 125424, 132839, 147669, 161017, 174364, 189195, 208475, 226271, 258898,
    303390, 361229, 509534, 1000000000
])

# Lift to drag ratio data, lift to drag ratio vs. AoA (degrees)
ld_ratio_data = np.array([
    
])

ld_vel_data = np.array([
    
])

def get_interpolated_drag_coefficient(vel):
    # vel = m/s
    if vel < cd_vel_data[-1] or vel > cd_vel_data[0]:
        raise ValueError(f"Velocity {vel} m/s is outside the data range [{cd_vel_data[-1]}, {cd_vel_data[0]}] m/s")
    
    # Since velocity_data is in descending order, reverse for np.interp
    return np.interp(vel, cd_vel_data[::-1], cd_drag_coeff_data[::-1])

def get_atmospheric_density(alt):
    # alt = meters, atmospheric density = kg/m^3
    surface_density = 0.0207 # kg/m^3 at sea level on Mars  
    scale_height = 11000  # Scale height in meters
    return surface_density * np.exp(-alt / scale_height)

def get_atmospheric_pressure(alt):
    # alt = meters, atmospheric pressure = Pa
    if (alt < altitude_data[1]):
        return 0.669*np.exp(-0.0000945*alt)*1000 # *1000 to convert to Pa

    return np.interp(alt, altitude_data, pressure_data)

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

