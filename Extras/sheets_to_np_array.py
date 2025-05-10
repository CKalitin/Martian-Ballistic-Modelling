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
    
    # Sort by velocity descending
    idx = np.argsort(-v_arr)
    return v_arr[idx], h_arr[idx]

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
466	12,022
521	13,388
621	14,208
721	14,754
832	14,754
943	14,754
1,021	14,208
1,098	14,208
1,176	14,208
1,243	13,934
1,320	13,934
1,398	13,934
1,476	13,934
1,553	13,934
1,631	13,934
1,709	13,934
1,786	13,934
1,864	14,208
1,942	14,208
2,008	14,208
2,086	14,208
2,164	14,481
2,241	14,481
2,308	14,754
2,386	14,754
2,463	15,027
2,530	15,301
2,607	15,574
2,685	15,847
2,752	15,847
2,829	15,847
2,907	16,120
2,974	16,393
3,051	16,667
3,129	16,940
3,196	16,940
3,273	17,486
3,340	17,760
3,417	18,033
3,484	18,306
3,562	18,579
3,628	18,852
3,706	19,399
3,773	19,399
3,839	19,945
3,917	20,492
3,983	20,765
4,061	21,038
4,128	21,585
4,194	21,858
4,261	22,404
4,338	22,951
4,405	23,224
4,472	23,770
4,538	24,317
4,605	24,863
4,682	25,410
4,738	25,956
4,816	26,503
4,871	27,322
4,949	27,596
5,126	30,055
5,193	31,148
5,259	31,967
5,326	33,333
5,381	34,426
5,448	35,792
5,503	37,158
5,548	38,798
5,581	40,437
5,614	42,077
5,637	43,989
5,659	45,902
5,670	47,814
5,681	49,454
5,703	51,366
5,692	53,825
5,681	55,738
5,670	57,377
5,659	59,290
5,648	60,929
5,648	62,842
5,648	64,754
5,648	66,667
5,659	68,306
5,670	70,219
5,681	71,858
5,681	73,770
5,614	77,049
5,603	78,689
5,603	80,601
5,603	82,514
5,592	84,426
5,592	86,066
5,592	87,978
5,592	89,891
5,592	91,803
5,592	93,716
5,581	95,628
5,581	97,541
5,581	99,180
5,592	101,093
5,592	103,005
5,581	104,645
5,592	106,557
5,592	108,470
5,592	110,383
5,592	112,295
5,603	113,934
5,603	115,847
5,603	117,760
5,603	119,672
5,614	121,585
5,614	123,224
    """
    
vel,alt = parse_vel_alt(raw_data)
print(format_for_python("Curiosity", vel, alt))