import numpy as np

# Data: velocity (m/s) and drag coefficient
cd_vel_data = np.array([
    100, 5.89, 5.88, 5.85, 5.80, 5.59, 5.40, 5.20, 5.00, 4.79, 4.59, 
    4.40, 4.19, 3.99, 3.79, 3.60, 3.40, 3.20, 3.00, 2.80, 2.60, 
    2.40, 2.19, 1.99, 1.79, 1.59, 1.40, 1.20, 1.00, 0.80, 0.62, 
    0.54, 0.49, 0.46, 0.43, 0.40, 0.37, 0.31, 0.26, 0.22, 0.19, 
    0.18, 0.15, 0.10, 0
])
cd_drag_coeff_data = np.array([
    1.75, 1.75, 1.71, 1.71, 1.72, 1.73, 1.73, 1.73, 1.73, 1.73, 1.73, 
    1.73, 1.73, 1.72, 1.71, 1.70, 1.69, 1.68, 1.67, 1.66, 1.65, 
    1.65, 1.64, 1.64, 1.63, 1.62, 1.61, 1.60, 1.59, 1.58, 1.58, 
    1.58, 1.58, 1.57, 1.57, 1.57, 1.57, 1.54, 1.42, 1.30, 1.22, 
    1.16, 1.10, 1.04, 1.00
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

def get_gravity_acc(alt):
    # alt = meters, gravity = m/s^2
    
    gravitational_constant = 6.67430e-11  # m^3 kg^-1 s^-2
    mars_radius = 3396200  # Mars radius in meters
    mars_mass = 6.4171e23  # Mars mass in kg
    
    distance = mars_radius + alt  # Distance from the center of Mars
    
    return gravitational_constant * (mars_mass) / (distance ** 2)

def get_drag_acc(alt, vel, area, mass):
    # alt = meters, vel = m/s, area = m^2, mass = kg
    return 1/mass * 0.5 * get_atmospheric_density(alt) * get_interpolated_drag_coefficient(vel) * area * vel**2

def get_drag_acc(mass, vel, area, drag_coeff, atm_density):
    # atm_density = kg/m^3, vel = m/s, area = m^2, drag_coeff = dimensionless
    return 1/mass * 0.5 * atm_density * drag_coeff * area * vel**2

