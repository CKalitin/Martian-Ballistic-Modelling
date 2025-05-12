import sim
import matplotlib.pyplot as plt
import numpy as np

file_path = 'Ballistic-Coefficient-Charts/'

# simulation parameters
time_step = 0.1
time_max = 100000
entry_altitude = 125000
entry_flight_path_angle = -15
entry_velocity = 6000
area = 1

# ballistic coefficients and corresponding masses
ballistic_coefficients = [1,3.7,10,52,100,196,400,1000,3700,10000,37000,100000]
masses = [bc * area for bc in ballistic_coefficients] 

# annotations
annotation_indexes = [3,5,6]
annotation_aoas = [0,20,30]
annotation_labels = ['Opportunity','Perseverance','Starship']

# sweep AoA from -25 to +25 in 5° increments
aoa_list = list(range(0, 90, 10))

# store terminal velocities for each AoA
results = {}

for aoa in aoa_list:
    print(f"\n=== Running AoA = {aoa} ===")
    tvs = []
    for mass in masses:
        sim_data = sim.simulate(
            time_step=time_step,
            time_max=time_max,
            mass=mass,
            area=area,
            aoa=aoa,
            entry_altitude=entry_altitude,
            entry_flight_path_angle=entry_flight_path_angle,
            entry_velocity=entry_velocity,
            verbose=False,
        )
        term_v = sim_data['velocities'][-1]
        tvs.append(term_v)
        print(f"  Mass={mass:.3f} kg -> V_terminal={term_v:.2f} m/s")
        sim.plot(sim_data, filename=f"{file_path}raw/BC_{(mass / area):.3f}_AoA_{aoa:+d}".replace('.', '-'), show=False)
    results[aoa] = np.array(tvs)

# print CSV-style summary
print("\nBallistic Coefficient," + ",".join(f"AoA{aoa:+d}" for aoa in aoa_list))
for i, bc in enumerate(ballistic_coefficients):
    row = [f"{results[aoa][i]:.2f}" for aoa in aoa_list]
    print(f"{bc}," + ",".join(row))

# combined log–log plot
plt.figure(figsize=(9.6, 5.4), dpi=200)
for aoa in aoa_list:
    plt.plot(ballistic_coefficients, results[aoa], marker='.', label=f"AoA {aoa}°")

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Ballistic Coefficient (kg/m²)')
plt.ylabel('Terminal Velocity (m/s)')
plt.title('Blunt Body Mars Entry Vehicle Terminal Velocity vs Ballistic Coefficient')
plt.grid(True, which='both', ls='--', alpha=0.5)

# annotate special points for one of the AoA series (e.g. AoA=0)
for idx, aoa, label in zip(annotation_indexes, annotation_aoas, annotation_labels):
    plt.annotate(label,
                 xy=(ballistic_coefficients[idx], results[aoa][idx]),
                 xytext=(ballistic_coefficients[idx]*1.4, results[aoa][idx]*0.6),
                 arrowprops=dict(arrowstyle='-|>',facecolor='black'),)

plt.legend(title='Angle of Attack')

textbox_text = f"""
    Christopher Kalitin 2025
    Entry Altitude: {round(entry_altitude/1000)} km
    Entry Velocity: {round(entry_velocity/1000)} km/s
    Entry Flight Path Angle: {entry_flight_path_angle}°
    Time Step: {time_step} s
    AoA is purely in the vertical plane.    
"""

plt.text(0.620, 0.325, textbox_text,
         transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle="round,pad=0.2", edgecolor='black', facecolor='white'))

plt.savefig(f"{file_path}terminal_velocity_vs_BC_t{time_step}_allAoA_max{aoa_list[-1]}_log.png")
plt.close()
