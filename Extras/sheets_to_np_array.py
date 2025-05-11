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
        v = int(parts[0])
        h = int(parts[1])
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
311	12,842
344	15,027
355	16,120
388	17,486
544	22,678
599	23,770
655	24,317
732	24,863
766	25,137
799	25,410
1,121	25,683
1,154	25,683
1,187	25,683
1,376	25,683
1,564	25,683
1,886	26,503
2,264	27,596
2,441	27,869
2,619	28,415
2,785	29,235
2,963	29,781
3,118	30,874
3,284	31,694
3,639	34,426
3,917	37,432
4,072	39,617
4,172	41,257
4,361	46,448
4,449	50,546
4,483	54,645
4,527	63,115
4,516	67,486
4,516	72,131
4,516	75,683
4,516	78,962
    """
    
vel,alt = parse_vel_alt(raw_data)
print(format_for_python("Viking", vel, alt))