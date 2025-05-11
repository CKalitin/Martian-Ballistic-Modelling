import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

separator = '\t'  # Default separator for TSV files

spacing = 5  # User-defined spacing

# Example usage with provided data (as strings for demonstration)
altitude_data = """0.000000	0.000000
5.000000	25669.166000
10.000000	51397.822000
15.000000	77184.882000
20.000000	103031.639000
25.000000	128928.083000
30.000000	154866.123000
35.000000	180843.456000
40.000000	206835.951000
45.000000	232799.507000
50.000000	258717.730000
55.000000	284467.884000
60.000000	309913.575000
65.000000	334751.361000
70.000000	358545.334000
75.000000	380626.496000
80.000000	400476.287000
85.000000	418157.460000
90.000000	433572.382000
95.000000	446941.987000
100.000000	458418.409000
105.000000	468450.825000
110.000000	477410.027000
115.000000	485380.582000
120.000000	492518.256000
125.000000	499024.797000
130.000000	505000.398000
135.000000	510516.675000
140.000000	515640.113000
145.000000	520433.409000
150.000000	524938.240000
155.000000	529200.515000
160.000000	533238.194000
165.000000	537064.792000
170.000000	540696.703000
175.000000	544142.826000
180.000000	547431.181000
185.000000	550574.536000
190.000000	553573.699000
195.000000	556455.296000
200.000000	559238.802000
205.000000	561930.611000
210.000000	564529.016000
215.000000	567032.320000
220.000000	569437.220000
225.000000	571742.221000
230.000000	573950.727000
235.000000	576058.951000
240.000000	577785.239000
245.000000	579024.011000
250.000000	579960.902000
255.000000	580716.550000
260.000000	581351.450000
265.000000	581886.451000
270.000000	582325.058000
275.000000	582675.574000
280.000000	582944.696000
285.000000	583141.535000
290.000000	583299.313000
295.000000	583442.303000
300.000000	583575.495000
305.000000	583705.596000
310.000000	583838.798000
315.000000	583966.788000
320.000000	584084.778000
325.000000	584192.162000
330.000000	584269.122000
335.000000	584306.385000
340.000000	584314.668000
345.000000	584325.082000
350.000000	584347.193000
355.000000	584382.910000
360.000000	584441.748000
365.000000	584568.525000
370.000000	584740.141000
375.000000	584904.606000
380.000000	584999.122000
385.000000	585037.092000
390.000000	585069.203000
395.000000	585114.617000
400.000000	585171.728000
405.000000	585242.142000
410.000000	585324.152000"""

velocity_data = """
0	125000
24.277078	91603.05344
42.772762	68702.29008
57.569309	50381.67939
71.132811	36641.22137
87.162404	25190.8397
95.793724	22900.76336
115.522454	20610.68702
134.018138	20610.68702
188.272145	20610.68702
199.369556	20610.68702
225.263514	18320.61069
241.293107	16030.53435
263.487928	13740.45802
293.081023	11450.38168
311.576708	9160.305344
328.839346	6870.229008
346.101985	4580.152672
362.131578	2290.076336
390.093871	0"""


def read_data(data_input, delimiter='\t', is_file=True):
    """
    Read data from a file or string, supporting TSV or CSV.
    
    Parameters:
    - data_input: File path (str) or data string (str).
    - delimiter: Separator ('\t' for TSV, ',' for CSV).
    - is_file: True if input is a file path, False if it's a string.
    
    Returns:
    - time, value: NumPy arrays for time and value (altitude or velocity).
    """
    if is_file:
        # Read from file
        df = pd.read_csv(data_input, delimiter=delimiter, header=None)
    else:
        # Read from string
        df = pd.read_csv(StringIO(data_input), delimiter=delimiter, header=None)
    
    # Convert to NumPy arrays
    time = df[0].to_numpy()
    value = df[1].to_numpy()
    return time, value

def unify_datasets(time_alt, alt, time_vel, vel, time_start=None, time_end=None, spacing=1.0):
    """
    Unify altitude and velocity datasets to a common time grid.
    
    Parameters:
    - time_alt, alt: NumPy arrays for altitude vs. time.
    - time_vel, vel: NumPy arrays for velocity vs. time.
    - time_start: Start of time grid (default: max of min times).
    - time_end: End of time grid (default: min of max times).
    - spacing: Time step for the grid (e.g., 1.0 for 1s intervals).
    
    Returns:
    - alt_interp, vel_interp: Interpolated altitude and velocity on common grid.
    """
    # Determine time range
    time_start = max(min(time_alt), min(time_vel)) if time_start is None else time_start
    time_end = min(max(time_alt), max(time_vel)) if time_end is None else time_end
    
    # Create common time grid
    time_common = np.arange(time_start, time_end + spacing, spacing)
    
    # Interpolate both datasets
    alt_interp = np.interp(time_common, time_alt, alt)
    vel_interp = np.interp(time_common, time_vel, vel)
    
    return time_common, alt_interp, vel_interp

def print_points(time, series1, series2):
    """
    Print interpolated time, altitude, and velocity points.
    
    Parameters:
    - time: Common time grid.
    - alt: Interpolated altitude values.
    - vel: Interpolated velocity values.
    """
    print("\nInterpolated Points (Time, series1, series2):")
    print(f"{'Time (s)':<12} {'series1':<20} {'series2':<15}")
    print("-" * 50)
    for t, a, v in zip(time, series1, series2):
        print(f"{t:<12.6f} {a:<20.6f} {v:<15.6f}")
        
def plot_alt_vs_vel(alt, vel, title="Altitude vs. Velocity"):
    """
    Plot altitude vs. velocity.
    
    Parameters:
    - alt, vel: Interpolated altitude and velocity arrays.
    - title: Plot title.
    """
    plt.plot(vel, alt, '-o', markersize=3)
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Altitude (m)')
    plt.title(title)
    plt.grid(True)
    plt.show()


# Read data (assuming string input for this example)
time_alt, alt = read_data(altitude_data, delimiter=separator, is_file=False)
time_vel, vel = read_data(velocity_data, delimiter=separator, is_file=False)

# Unify datasets with custom time spacing (e.g., 1s)
time_common, alt_interp, vel_interp = unify_datasets(time_alt, alt, time_vel, vel, spacing=spacing)

#subtract 545.39 from time common to align with simulation time
#time_common -= 545.39

# Plot
#plot_alt_vs_vel(alt_interp, vel_interp, title=f"Altitude vs. Velocity (Time Spacing: {spacing}s)")

# Print points
print_points(time_common,  alt_interp, vel_interp)