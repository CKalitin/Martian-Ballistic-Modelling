import sim

# Curiosity
sim_data = sim.simulate(
    time_step=0.01,
    time_max=1000,
    mass=3300,
    area=15.9,
    entry_altitude=125000,
    entry_flight_path_angle=-14,
    entry_velocity=5800,
    print_debug=False,
)
sim.plot(sim_data, filename="backtest/Curiosity-Entry.png")