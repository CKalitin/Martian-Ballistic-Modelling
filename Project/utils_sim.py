import numpy as np
import utils_data

class SimUtils:
    def __init__(self, sim_parameters):
        self.sim_parameters = sim_parameters

    def get_interpolated_drag_coefficient(self, vel):
        # vel = m/s
        if vel < self.sim_parameters.cd_vel_data[-1] or vel > self.sim_parameters.cd_vel_data[0]:
            raise ValueError(f"Velocity {vel} m/s is outside the data range [{self.sim_parameters.cd_vel_data[-1]}, {self.sim_parameters.cd_vel_data[0]}] m/s")
        
        # Since velocity_data is in descending order, reverse for np.interp
        return np.interp(vel, self.sim_parameters.cd_vel_data[::-1], self.sim_parameters.cd_drag_coeff_data[::-1])

    def get_interpolated_lift_to_drag_ratio(self, aoa):
        # aoa = degrees
        if aoa < self.sim_parameters.ld_deg_data[0] or aoa > self.sim_parameters.ld_deg_data[-1]:
            raise ValueError(f"Angle of attack {aoa} degrees is outside the data range [{self.sim_parameters.ld_deg_data[-1]}, {self.sim_parameters.ld_deg_data[0]}] degrees")

        return np.interp(aoa, self.sim_parameters.ld_deg_data, self.sim_parameters.ld_ratio_data)

    def get_numpy_aoa_list(self, aoa_list):
        # aoa_list = [(alt, aoa), ...], alt = meters, aoa = degrees
        out = np.array(aoa_list)
        # reverse the order of both columns to be in descending order
        out[:, 0] = out[:, 0][::-1]
        out[:, 1] = out[:, 1][::-1]
        return out

    def get_interpolated_aoa(self, numpy_aoa_list, vel):
        return np.interp(vel, numpy_aoa_list[:,0], numpy_aoa_list[:,1])

    def get_atmospheric_pressure(self, alt):
        # alt = meters, atmospheric pressure = Pa
        if (alt < self.sim_parameters.pressure_alt_data[0]):
            # Note this is only for Mars
            return 699*np.exp(-0.0000945*alt)

        return np.interp(alt, self.sim_parameters.pressure_alt_data, self.sim_parameters.pressure_data)

    def get_temperature(self, alt):
        # alt = meters, temperature = K
        return np.interp(alt, self.sim_parameters.temp_alt_data, self.sim_parameters.temp_data)

    def get_atmospheric_density(self, alt, pressure=None, temperature=None):
        # alt = meters, pressure = Pa, temperature = K, atmospheric density = kg/m^3
        if pressure is None:
            pressure = self.get_atmospheric_pressure(alt)
        if temperature is None:
            temperature = self.get_temperature(alt)
        
        r = utils_data.UNIVERSAL_GAS_CONSTANT / self.sim_parameters.atmospheric_molar_mass * 1000 # About 192.1 g/mol
    
        return pressure / (r * temperature)  # Ideal gas law: density = pressure / (R * T)

    def get_gravity_acc(self, radial_distance):
        # radial_distance = distance from the center of Mars (m)
        return -utils_data.GRAVITATIONAL_CONSTANT * self.sim_parameters.body_mass / radial_distance**2

    def get_drag_acc(self, mass, vel_net, area, atm_density):
        # mass = kg, vel_net = m/s, area = m^2, atm_density = kg/m^3
        drag_coeff = self.get_interpolated_drag_coefficient(vel_net)
        return -0.5 * atm_density * vel_net**2 * drag_coeff * area / mass

    def get_lift_acc(self, drag_acc, aoa):
        # drag_acc = m/s^2, flight_path_angle = degrees
        ld = self.get_interpolated_lift_to_drag_ratio(aoa)
        return -drag_acc * ld if ld != 0 else 0
