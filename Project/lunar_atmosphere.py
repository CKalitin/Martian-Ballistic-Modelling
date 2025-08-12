import sim_polar
import utils_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BOLTZMANN_CONSTANT = 1.380649e-23 # J/K
ATMOSPHERIC_TEMPERATURE = 250 # K
MEAN_MOLAR_MASS = 0.04 # kg/mol
GRAVITATIONAL_ACCELERATION = 1.62 # m/s²
AVOGADRO_NUMBER = 6.02214076e23 # mol⁻¹

# H = k * T / (g * M / N), there M/N is molecular mass derived from molar mass
atmospheric_scale_height = BOLTZMANN_CONSTANT * ATMOSPHERIC_TEMPERATURE / (GRAVITATIONAL_ACCELERATION * MEAN_MOLAR_MASS / AVOGADRO_NUMBER)

def set_body_pressure(sim_parameters: utils_data.SimParameters, surface_pressure):
    def alt_to_pressure(alt, surface_pressure):
        # p(h) = p0 * exp(-h / H)
        return surface_pressure * np.exp(-alt / atmospheric_scale_height)

    altitudes = np.concatenate([
        np.arange(0, 2100, 100),            # 0 to 2,000 in 100 increments
        np.arange(3000, 51000, 1000),       # 3,000 to 50,000 in 1,000 increments
        np.arange(60000, 210000, 10000),    # 60,000 to 200,000 in 10,000 increments
        np.array([1000000000])              # 1,000,000,000
    ])
    
    sim_parameters.pressure_data = alt_to_pressure(altitudes, surface_pressure)  # Pa
    sim_parameters.pressure_alt_data = altitudes  # m
    
    def plot():
        # Plot Atmosphere using matplotlib
        plt.figure()
        plt.semilogy(sim_parameters.pressure_data, sim_parameters.pressure_alt_data)
        plt.title('Lunar Atmosphere Pressure vs Altitude (Example 1e5 Pa surface pressure)')
        plt.xlabel('Pressure (Pa)')
        plt.ylabel('Altitude (m)')
        plt.ylim(0, 1e6)  # Set y-axis min to 0 and max to 1,000,000
        plt.grid()

        # Format y-axis to display altitude in plain numbers (not scientific notation)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

        plt.show()
    
    #plot()

def set_body_parameters(sim_parameters: utils_data.SimParameters):
    sim_parameters.body_radius = 1737400  # Lunar radius in meters
    sim_parameters.body_mass = 7.34767309e22  # Lunar mass in kg
    sim_parameters.atmospheric_molar_mass = 40.0  # g/mol (for Moon's atmosphere, oxygen rich so higher)

    sim_parameters.temp_data = np.array([250, 250]) # K, constant temp
    sim_parameters.temp_alt_data = np.array([-10000000, 10000000]) # m

def set_aerodynamic_parameters(sim_parameters: utils_data.SimParameters, vehicle='shuttle'):
    if vehicle == 'shuttle':
        # Coefficient of drag data, drag coefficient vs. velocity (m/s)
        # Using the same paper, assuming 30 degree constant angle of attack and mach 4ish because I'm not putting that much effort into this
        # https://aviation.stackexchange.com/questions/35305/what-was-the-space-shuttles-glide-ratio
        sim_parameters.cd_drag_coeff_data = np.array([0.42, 0.42])
        sim_parameters.cd_vel_data = np.array([100000, -100000])  # m/s
        
        # Lift to drag ratio data, lift to drag ratio vs. AoA (degrees)
        # This is actually dependent on velocity too, which I missed in my first simulation. I'll go with Mach=2 line
        # https://aviation.stackexchange.com/questions/35305/what-was-the-space-shuttles-glide-ratio
        sim_parameters.ld_ratio_data = np.array([
            -1.2, -1.8, -2.4, -2.8, -2.8, -1.5, 0, 1.5, 2.8, 2.8, 2.4, 1.8, 1.2
        ])
        sim_parameters.ld_deg_data = np.array([
            -45, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 45
        ])
        
    # All original values for a Mars blunt body (Used Pheonix lander data)
    elif vehicle == 'blunt_body' or vehicle == 'starship':
        # Coefficient of drag data, drag coefficient vs. velocity (m/s)
        sim_parameters.cd_drag_coeff_data = np.array([
            1.73, 1.72, 1.71, 1.71, 1.72, 1.73, 1.73, 1.73, 1.73, 1.73,
            1.73, 1.73, 1.73, 1.72, 1.71, 1.70, 1.69, 1.68, 1.67, 1.66,
            1.65, 1.65, 1.64, 1.64, 1.63, 1.62, 1.61, 1.60, 1.59, 1.58,
            1.58, 1.58, 1.58, 1.57, 1.57, 1.57, 1.57, 1.54, 1.42, 1.30,
            1.22, 1.16, 1.10, 1.04, 1.00
        ])

        sim_parameters.cd_vel_data = np.array([
            100000, 5890, 5880, 5850, 5800, 5590, 5400, 5200, 5000, 4790,
            4590, 4400, 4190, 3990, 3790, 3600, 3400, 3200, 3000, 2800,
            2600, 2400, 2190, 1990, 1790, 1590, 1400, 1200, 1000, 800,
            620, 540, 490, 460, 430, 400, 370, 310, 260, 220,
            190, 180, 150, 100, 0, 
        ])

        # Lift to drag ratio data, lift to drag ratio vs. AoA (degrees)
        sim_parameters.ld_ratio_data = np.array([
            -1.5,-0.265,-0.175,-0.095,-0.065,-0.035,0,0.035,0.065,0.095,0.175,0.265,1.5
        ])

        sim_parameters.ld_deg_data = np.array([
            -90,-16,-11,-6,-4,-2,0,2,4,6,11,16,90
    ])

def test():
    sim_parameters = utils_data.SimParameters(body='Lunar')
    set_body_parameters(sim_parameters)
    set_body_pressure(sim_parameters, surface_pressure=1e-5)
    set_aerodynamic_parameters(sim_parameters, vehicle='blunt_body')

    data, params = sim_polar.simulate(
        sim_parameters=sim_parameters,
        mass=1000,
        area=10,
        entry_altitude=200000,
        entry_flight_path_angle=0,
        entry_velocity=1600,
        aoa_function=20,
        time_step=0.1,
        time_max=10000,
        verbose=False,
    )

    # 1591 m/s is orbital velocity at 200 km.
    # 1600 m/s is 243x200 km, if starting at 200 km.

    sim_polar.plot(sim_parameters, data, params, title="Lunar Entry Simulation", file_name="lunar_test.png", show=True)
    
def sim_atmospheres(surface_atmospheric_pressures, file_path="Lunar Atmosphere/"):
    vehicles = ['shuttle', 'blunt_body', 'starship']
    results = {vehicle: [] for vehicle in vehicles}
    params = None
    for vehicle in vehicles:
        sim_parameters = utils_data.SimParameters(body='Lunar')
        set_body_parameters(sim_parameters)
        set_aerodynamic_parameters(sim_parameters, vehicle=vehicle)

        for atm_p in surface_atmospheric_pressures:
            print(f"=== Running {vehicle} with surface pressure {atm_p} Pa ===")
            set_body_pressure(sim_parameters, atm_p)

            aoa = 0 if vehicle == 'blunt_body' else 10 # Constnat 10 AoA for shuttle

            mass = 4000 if vehicle == 'starship' else 1000

            data, params = sim_polar.simulate(
                sim_parameters=sim_parameters,
                mass=mass,
                area=10,
                entry_altitude=200000,
                entry_flight_path_angle=-5,
                entry_velocity=1600,
                aoa_function=aoa,
                time_step=0.1,
                time_max=10000,
                verbose=False,
            )

            sim_polar.plot(
                sim_parameters,
                data, params,
                title=f"{vehicle} Entry Simulation at {atm_p} Pa",
                file_name=f"{file_path}/lunar_{vehicle}_atm_{atm_p}.png",
                show=False
            )
            results[vehicle].append(data.v_net[-1])  # Store terminal velocity
    
    # combined log–log plot
    plt.figure(figsize=(9.6, 5.4), dpi=200)
    for vehicle, label in zip(results.keys(), ['Shuttle 100 kg/m² 10° AoA', 'Blunt Body 100 kg/m² 0° AoA', 'Starship 400 kg/m² 30° AoA']):
        plt.plot(surface_atmospheric_pressures, results[vehicle], marker='.', label=label)
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlabel('Surface Atmospheric Pressure (Pa)')
    plt.ylabel('Impact Velocity (m/s)')
    plt.title('Lunar Entry Vehicle Impact Velocity vs Lunar Surface Atmospheric Pressure')
    plt.grid(True, which='both', ls='--', alpha=0.5)

    plt.legend(title='Vehicle')

    textbox_text = f"""
        Christopher Kalitin 2025
        Entry Altitude: {round(params['entry_altitude']/1000)} km
        Entry Velocity: {round(params['entry_velocity']/1000, 2)} km/s
        Entry Flight Path Angle: {params['entry_flight_path_angle']}°
        Time Step: {params['time_step']} s
        Atm. Scale Height: 32.1 km       
    """

    plt.text(0.640, 0.745, textbox_text,
            transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle="round,pad=0.2", edgecolor='black', facecolor='white'))

    plt.savefig(f"{file_path}impact_velocity_vs_atm_t{params['time_step']}_linear.png")
    plt.close()

atms = np.array([10, 50, 100, 500, 1000, 2000])  # Pa
sim_atmospheres(atms, file_path="Lunar Atmosphere/")