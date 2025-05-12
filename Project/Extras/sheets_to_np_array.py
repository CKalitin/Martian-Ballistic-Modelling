# This converts the Google Sheets velocity, altitude columns into the numpy array required

import numpy as np

def parse_vel_alt(data_str):
    """
    Parse a tab‑or‑space‑delimited multiline string with header
    Velocity (m/s)   Altitude (m)
    into two sorted numpy arrays (velocities desc, same order altitudes).
    """
    lines = data_str.strip().splitlines()
    # Skip header
    data_lines = [l for l in lines if not l.lower().startswith('velocity')]
    
    velocities = []
    alts       = []
    for line in data_lines:
        # split on whitespace or tabs
        parts = line.strip().replace(',', '').split()
        if len(parts) < 2:
            continue
        v = int(float(parts[0]))
        h = int(float(parts[1]))
        velocities.append(v)
        alts.append(h)
    
    # Convert to numpy arrays
    v_arr = np.array(velocities, dtype=int)
    h_arr = np.array(alts,       dtype=int)
    
    return v_arr, h_arr

def format_for_python(name, v_arr, h_arr):
    """
    Return a formatted Python snippet defining:
      <name>_vel_alt = [
        np.array([...]),
        np.array([...]),
        "<Name>"
      ]
    """
    lines = []
    lines.append(f"# {name} Entry:")
    lines.append(f"{name.lower()} = [")
    # velocities
    vel_list = ", ".join(str(x) for x in v_arr)
    lines.append("    # Velocity (m/s)")
    lines.append("    np.array([")
    lines.append("    " + vel_list)
    lines.append("    ]),")
    # altitudes
    alt_list = ", ".join(str(x) for x in h_arr)
    lines.append("    # Altitude (m)")
    lines.append("    np.array([")
    lines.append("    " + alt_list)
    lines.append("    ]),")
    # name
    lines.append(f'    "{name}"')
    lines.append("]")
    return "\n".join(lines)

if __name__ == "__main__":
    # paste your sheet output into this triple‑quoted string:
    raw_data = """Velocity (m/s)	Altitude (m)
0.000000, 0.000000
73.529412, 535.714286
147.058824, 803.571429
275.735294, 1205.357143
386.029412, 1741.071429
514.705882, 2410.714286
533.088235, 4419.642857
569.852941, 6562.500000
606.617647, 7633.928571
643.382353, 8839.285714
680.147059, 9508.928571
753.676471, 9910.714286
845.588235, 9642.857143
919.117647, 8973.214286
1029.411765, 7901.785714
1158.088235, 6696.428571
1286.764706, 5758.928571
1397.058824, 5089.285714
1507.352941, 4687.500000
1599.264706, 4553.571429
1709.558824, 4419.642857
1819.852941, 4553.571429
1985.294118, 4955.357143
2132.352941, 5491.071429
2316.176471, 6428.571429
2481.617647, 7366.071429
2628.676471, 8437.500000
2812.500000, 9910.714286
2996.323529, 11785.714286
3125.000000, 13258.928571
3253.676471, 14732.142857
3382.352941, 16071.428571
3529.411765, 17812.500000
3639.705882, 19017.857143
3750.000000, 20223.214286
3878.676471, 21562.500000
3970.588235, 22500.000000
4080.882353, 23705.357143
4209.558824, 24910.714286
4393.382353, 26651.785714
4558.823529, 28125.000000
4705.882353, 29330.357143
4852.941176, 30401.785714
5091.911765, 32008.928571
5257.352941, 32946.428571
5422.794118, 33750.000000
5625.000000, 34419.642857
5845.588235, 34955.357143
6066.176471, 35357.142857
6305.147059, 35758.928571
6488.970588, 36160.714286
6691.176471, 36830.357143
6875.000000, 37767.857143
7022.058824, 38839.285714
7169.117647, 40446.428571
7242.647059, 41785.714286
7297.794118, 43125.000000
7352.941176, 44732.142857
7389.705882, 46473.214286
7408.088235, 47410.714286
7426.470588, 48883.928571
7444.852941, 50491.071429
7463.235294, 52366.071429
7463.235294, 54776.785714
7481.617647, 58125.000000
7481.617647, 59866.071429
    """
    
vel,alt = parse_vel_alt(raw_data)
print(format_for_python("Starship", vel, alt))